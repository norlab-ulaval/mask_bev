from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import nn
from torchmetrics.detection import MeanAveragePrecision

from mask_bev.evaluation.average_precision import batched_mask_iou
from mask_bev.evaluation.detection_metric import BinaryClassifMapMetric, MeanIoU
from mask_bev.models.networks.mask2former_head.mask2former_head import Mask2FormerHead
from mask_bev.utils.config import Config


class MaskBevPanopticHead(nn.Module):
    def __init__(self, in_channels: List[int], feat_channels: int, out_channels: int, num_queries: int,
                 num_classes: int, reverse_class_weights: bool = False, predict_height: bool = False):
        super().__init__()
        config = self._get_config(num_classes, 0, num_queries, in_channels, feat_channels, out_channels,
                                  reverse_class_weights)
        self._num_classes = num_classes
        self._predict_height = predict_height
        self._panoptic_head = Mask2FormerHead(**config, predict_height=self._predict_height)
        self._panoptic_head.init_weights()

    def forward(self, x):
        img_meta = [Config(dict(metainfo={})) for _ in range(x[0].shape[0])]
        return self._panoptic_head.forward(x, img_meta)

    def loss(self, cls, masks, label_gt, masks_gt, heights_pred, heights_gt):
        img_meta = [{} for i in range(cls[0].shape[0])]
        return self._panoptic_head.loss(cls, masks, label_gt, masks_gt, img_meta, heights_pred, heights_gt)

    def update_mAP_metrics(self, layer_index: int, pred_cls, pred_masks, labels_gt, masks_gt,
                           cls_metric: BinaryClassifMapMetric,
                           map_metric: MeanAveragePrecision, mIoU_metric: MeanIoU):
        """
        Computes the mAP for the given batch
        :param layer_index: index of the layer to use for the prediction
        :param pred_cls: class predictions (list of tensors (B, N, C), one item per decoder layer)
        :param pred_masks: mask predictions (list of tensors (B, N, H, W), one item per decoder layer)
        :param labels_gt: labels ground truth (B, N)
        :param masks_gt: masks ground truth (B, N, H, W)
        :param cls_metric: class metric
        :param mask_metric: mask metric
        :param mIoU_metric: mIoU metric
        :return: mAP
        """
        # Only keep predictions for the `layer_index` layer
        pred_cls = pred_cls[layer_index]
        pred_masks = pred_masks[layer_index]

        # Build gt instances and img meta
        batch_size = pred_cls.size(0)
        img_meta = [{} for _ in range(batch_size)]

        # Compute metrics per sample
        for sample_idx in range(batch_size):
            batch_pred_cls = pred_cls[sample_idx]
            batch_pred_masks = pred_masks[sample_idx]
            batch_labels_gt = labels_gt[sample_idx]
            batch_masks_gt = masks_gt[sample_idx]
            batch_gt_instances = InstanceData(labels=batch_labels_gt, masks=batch_masks_gt)
            batch_img_meta = img_meta[sample_idx]

            # Get targets
            labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds, sampling_result = self._panoptic_head._get_targets_single(
                batch_pred_cls, batch_pred_masks, batch_gt_instances, batch_img_meta, None)

            # TODO support multiple cls
            # TODO mAP per difficulty
            evaluated_class = 0
            pred_softmax = batch_pred_cls.softmax(dim=-1)
            y_scores = pred_softmax[:, evaluated_class]
            y_pred = batch_pred_cls.argmax(dim=-1)
            y_true = labels

            # Classification metric
            cls_metric.update(y_scores, y_true)

            # Upscale masks to target dim
            batch_pred_masks_to_target_dim = F.interpolate(batch_pred_masks.unsqueeze(1),
                                                           mask_targets.shape[1:], mode='bilinear',
                                                           align_corners=False).squeeze(1)
            batch_pred_masks_to_target_dim_pos = batch_pred_masks_to_target_dim[pos_inds]
            batch_pred_masks_to_target_dim_pos = torch.sigmoid(batch_pred_masks_to_target_dim_pos) > 0.5

            # IoU metric
            ious = batched_mask_iou(mask_targets, batch_pred_masks_to_target_dim_pos)
            mIoU_metric.update(ious)

            # mAP metric
            batch_pred_masks_to_target_dim = (torch.sigmoid(batch_pred_masks_to_target_dim) > 0.5)
            preds = [dict(boxes=None, scores=y_scores, labels=y_pred, masks=batch_pred_masks_to_target_dim)]
            target = [dict(boxes=None, labels=batch_labels_gt, masks=batch_masks_gt.to(torch.bool))]
            map_metric.update(preds, target)

    def _get_config(self, num_things_classes, num_stuff_classes, num_queries, in_channels, head_feat_channels,
                    head_out_channels, reverse_class_weights):
        num_classes = num_things_classes + num_stuff_classes
        class_weights = [1.0] * num_classes + [0.1]
        if reverse_class_weights:
            class_weights = list(reversed(class_weights))

        num_transformer_feat_level = 3
        num_heads = 8

        return Config(dict(
            type='Mask2FormerHead',
            in_channels=in_channels,
            strides=[4, 8, 16, 32],
            feat_channels=head_feat_channels,
            out_channels=head_out_channels,
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            num_queries=num_queries,
            num_transformer_feat_level=num_transformer_feat_level,
            align_corners=False,
            pixel_decoder=dict(
                type='mmdet.MSDeformAttnPixelDecoder',
                num_outs=num_transformer_feat_level,
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU'),
                encoder=dict(  # DeformableDetrTransformerEncoder
                    num_layers=6,
                    layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                        self_attn_cfg=dict(  # MultiScaleDeformableAttention
                            embed_dims=head_feat_channels,
                            num_heads=num_heads,
                            num_levels=num_transformer_feat_level,
                            num_points=4,
                            im2col_step=64,
                            dropout=0.0,
                            batch_first=True,
                            norm_cfg=None,
                            init_cfg=None),
                        ffn_cfg=dict(
                            embed_dims=head_feat_channels,
                            feedforward_channels=1024,
                            num_fcs=2,
                            ffn_drop=0.0,
                            act_cfg=dict(type='ReLU', inplace=True))),
                    init_cfg=None),
                positional_encoding=dict(  # SinePositionalEncoding
                    num_feats=head_feat_channels // 2, normalize=True),
                init_cfg=None),
            enforce_decoder_input_project=False,
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=head_feat_channels // 2, normalize=True),
            transformer_decoder=dict(  # Mask2FormerTransformerDecoder
                return_intermediate=True,
                num_layers=9,
                layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                    self_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=head_feat_channels,
                        num_heads=num_heads,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    cross_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=head_feat_channels,
                        num_heads=num_heads,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        dropout_layer=None,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=head_feat_channels,
                        feedforward_channels=2048,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True)),
                init_cfg=None),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=class_weights),
            loss_mask=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0),
            loss_dice=dict(
                type='mmdet.DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0),
            train_cfg=dict(
                num_points=12544,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.ClassificationCost', weight=2.0),
                        dict(
                            type='mmdet.CrossEntropyLossCost',
                            weight=5.0,
                            use_sigmoid=True),
                        dict(
                            type='mmdet.DiceCost',
                            weight=5.0,
                            pred_act=True,
                            eps=1.0)
                    ]),
                sampler=dict(type='mmdet.MaskPseudoSampler'))),
            test_cfg=dict(mode='whole'))

        # return Config(dict(
        #     in_channels=in_channels,  # pass to pixel_decoder inside
        #     strides=[4, 8, 16, 32],
        #     feat_channels=head_feat_channels,
        #     out_channels=head_out_channels,
        #     num_things_classes=num_things_classes,
        #     num_stuff_classes=num_stuff_classes,
        #     num_queries=num_queries,
        #     num_transformer_feat_level=num_transformer_feat_level,
        #     pixel_decoder=dict(
        #         type='MSDeformAttnPixelDecoder',
        #         num_outs=num_transformer_feat_level,
        #         norm_cfg=dict(type='GN', num_groups=32),
        #         act_cfg=dict(type='ReLU'),
        #         encoder=dict(
        #             type='DetrTransformerEncoder',
        #             num_layers=6,
        #             transformerlayers=dict(
        #                 type='BaseTransformerLayer',
        #                 attn_cfgs=dict(
        #                     type='MultiScaleDeformableAttention',
        #                     embed_dims=head_feat_channels,
        #                     num_heads=num_heads,
        #                     num_levels=num_transformer_feat_level,
        #                     num_points=4,
        #                     im2col_step=64,
        #                     dropout=0.0,
        #                     batch_first=False,
        #                     norm_cfg=None,
        #                     init_cfg=None),
        #                 ffn_cfgs=dict(
        #                     type='FFN',
        #                     embed_dims=head_feat_channels,
        #                     feedforward_channels=1024,
        #                     num_fcs=2,
        #                     ffn_drop=0.0,
        #                     act_cfg=dict(type='ReLU', inplace=True)),
        #                 operation_order=('self_attn', 'norm', 'ffn', 'norm')),
        #             init_cfg=None),
        #         positional_encoding=dict(
        #             type='SinePositionalEncoding', num_feats=head_feat_channels // 2, normalize=True),
        #         init_cfg=None),
        #     enforce_decoder_input_project=False,
        #     positional_encoding=dict(
        #         type='SinePositionalEncoding', num_feats=head_feat_channels // 2, normalize=True),
        #     transformer_decoder=dict(
        #         type='DetrTransformerDecoder',
        #         return_intermediate=True,
        #         num_layers=9,
        #         transformerlayers=dict(
        #             type='DetrTransformerDecoderLayer',
        #             attn_cfgs=dict(
        #                 type='MultiheadAttention',
        #                 embed_dims=head_feat_channels,
        #                 num_heads=num_heads,
        #                 attn_drop=0.0,
        #                 proj_drop=0.0,
        #                 dropout_layer=None,
        #                 batch_first=False),
        #             ffn_cfgs=dict(
        #                 embed_dims=head_feat_channels,
        #                 feedforward_channels=2048,
        #                 num_fcs=2,
        #                 act_cfg=dict(type='ReLU', inplace=True),
        #                 ffn_drop=0.0,
        #                 dropout_layer=None,
        #                 add_identity=True),
        #             feedforward_channels=2048,
        #             operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
        #                              'ffn', 'norm')),
        #         init_cfg=None),
        #     loss_cls=dict(
        #         type='CrossEntropyLoss',
        #         use_sigmoid=False,
        #         loss_weight=2.0,
        #         reduction='mean',
        #         class_weight=class_weights),
        #     loss_height=dict(
        #         type='CrossEntropyLoss',
        #         use_sigmoid=False,
        #         loss_weight=2.0,
        #         reduction='mean'),
        #     loss_mask=dict(
        #         type='CrossEntropyLoss',
        #         use_sigmoid=True,
        #         reduction='mean',
        #         loss_weight=5.0),
        #     loss_dice=dict(
        #         type='DiceLoss',
        #         use_sigmoid=True,
        #         activate=True,
        #         reduction='mean',
        #         naive_dice=True,
        #         eps=1.0,
        #         loss_weight=5.0)),
        #     train_cfg=dict(
        #         num_points=12544,
        #         oversample_ratio=3.0,
        #         importance_sample_ratio=0.75,
        #         assigner=dict(
        #             type='MaskHungarianAssigner',
        #             cls_cost=dict(type='ClassificationCost', weight=2.0),
        #             mask_cost=dict(
        #                 type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
        #             dice_cost=dict(
        #                 type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        #         sampler=dict(type='MaskPseudoSampler')),
        #     # test_cfg=dict(
        #     #     panoptic_on=True,
        #     #     # For now, the dataset does not support
        #     #     # evaluating semantic segmentation metric.
        #     #     semantic_on=False,
        #     #     instance_on=True,
        #     #     # max_per_image is for instance segmentation.
        #     #     max_per_image=100,
        #     #     iou_thr=0.8,
        #     #     # In Mask2Former's panoptic postprocessing,
        #     #     # it will filter mask area where score is less than 0.5 .
        #     #     filter_low_score=True)
        # )
