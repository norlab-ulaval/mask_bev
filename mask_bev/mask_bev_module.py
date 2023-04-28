import copy
import pathlib
from typing import Union, Dict, Any, Optional

import pytorch_lightning as pl
import torch
import torch_optimizer
from mask_bev.models.backbones.mask_bev_backbone import MaskBevBackbone
from mask_bev.models.encoders.mask_bev_encoders import MaskBevEncoder, EncodingType
from mask_bev.models.head.mask_bev_panoptic_head import MaskBevPanopticHead
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from mask_bev.evaluation.detection_metric import DetectionMapMetric, BinaryClassifMapMetric, MeanIoU, MaskArea
from mask_bev.models.training_types import OptimizerType, LrSchedulerType

# TODO test global vs local (build a scene without global info)
# TODO analyse min points for detection with global info
# TODO test hydra attention
"""
def hydra(q, k, v):
    # q, k, and v should all be tensors of shape
    # [batch, tokens, features]
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    kv = (k * v).sum(dim=-2, keepdim=True)
    out = q * kv
    return out
"""


# TODO Mask dataset: flip all cars along its x-axis to have both sides (behind the curtain)
# TODO Mask from bounding boxes
# TODO increase num layers in decoder
# TODO lr warmup
class MaskBevModule(pl.LightningModule):
    def __init__(self, x_range: (int, int), y_range: (int, int), z_range: (int, int), voxel_size: float,
                 num_queries: int, max_num_points: int, encoder_feat_channels: [int], backbone_embed_dim: int,
                 head_feat_channels: int, head_out_channels: int, optimiser_type: Union[OptimizerType, str], lr: float,
                 weight_decay: float, lr_schedulers_type: Union[LrSchedulerType, str], differential_lr: bool,
                 differential_lr_scaling: float, encoder_encoding_type: str = EncodingType.Vanilla,
                 encoder_fourier_enc_group: int = 1, backbone_patch_size: int = 4, backbone_window_size: int = 10,
                 backbone_strides: (int, int, int, int) = (4, 2, 2, 2), backbone_use_abs_emb: bool = True,
                 backbone_swap_dims: bool = False, head_reverse_class_weights: bool = False, head_num_classes: int = 1,
                 pc_point_dim: int = 4, predict_heights: bool = False, **kwargs):
        """
        MaskBev base network

        References to mmdetection configs:
        - mmdetection3d/configs/_base_/models/hv_second_secfpn_kitti.py
        - mmdetection/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py
        - mmdetection/configs/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco-panoptic.py
        """
        super().__init__()
        self._optimiser_type = optimiser_type
        self._lr = lr
        self._weight_decay = weight_decay
        self._lr_schedulers_type = lr_schedulers_type
        self._differential_lr = differential_lr
        self._differential_lr_scaling = differential_lr_scaling
        self._predict_heights = predict_heights

        # encoder
        voxel_size_z = z_range[1] - z_range[0]

        # head
        head_in_dims = [2 ** i * backbone_embed_dim for i in range(4)]
        # head_in_dims = [256, 512, 1024, 2048]

        num_voxel_x = int((x_range[1] - x_range[0]) / voxel_size)
        num_voxel_y = int((y_range[1] - y_range[0]) / voxel_size)
        img_size = (num_voxel_x, num_voxel_y)

        self._encoder = MaskBevEncoder(encoder_feat_channels, x_range, y_range, z_range, voxel_size,
                                       voxel_size, voxel_size_z, max_num_points, encoder_encoding_type,
                                       encoder_fourier_enc_group, encoder_params=dict(with_distance=True),
                                       pc_point_dim=pc_point_dim)
        self._backbone = MaskBevBackbone(img_size, encoder_feat_channels[-1], backbone_embed_dim, backbone_patch_size,
                                         backbone_window_size, backbone_strides, backbone_use_abs_emb,
                                         backbone_swap_dims)
        self._panoptic_head = MaskBevPanopticHead(head_in_dims, head_feat_channels, head_out_channels, num_queries,
                                                  head_num_classes, head_reverse_class_weights, predict_heights)

        self._train_mIoU = MeanIoU()
        self._train_cls_map = BinaryClassifMapMetric()
        self._train_height_map = DetectionMapMetric() if self._predict_heights else None
        # self._train_cls_map = DetectionMapMetric()
        self._train_mask_map_all = [DetectionMapMetric() for _ in range(10)]
        self._train_mask_map = DetectionMapMetric()
        self._train_easy_map = DetectionMapMetric()
        self._train_moderate_map = DetectionMapMetric()
        self._train_hard_map = DetectionMapMetric()

        self._val_mIoU = MeanIoU()
        self._val_mIoU_points = MeanIoU()
        self._val_cls_map = BinaryClassifMapMetric()
        self._val_height_map = DetectionMapMetric() if self._predict_heights else None
        # self._val_cls_map = DetectionMapMetric()
        self._val_mask_map_all = [DetectionMapMetric().to('cuda') for _ in range(10)]
        self._val_mask_area = MaskArea()
        self._val_mask_map = DetectionMapMetric()
        self._val_easy_map = DetectionMapMetric()
        self._val_moderate_map = DetectionMapMetric()
        self._val_hard_map = DetectionMapMetric()

        self.save_hyperparameters()

    @staticmethod
    def from_config(config: Dict, exp_name: str,
                    checkpoint_folder_path: Optional[pathlib.Path] = None) -> 'MaskBevModule':
        """
        Load a model from a config dictionary
        :param config: dictionary of all parameters
        :param exp_name: experience name, used for checkpointing and logging
        :param checkpoint_folder_path: path to the root checkpoint folder, used only if 'checkpoint' is 'last'
        :return: MaskBevModule
        """
        seed = config['seed']
        pl.seed_everything(seed)

        checkpoint = config.get('checkpoint', None)
        if checkpoint is not None:
            if checkpoint == 'last':
                checkpoint_path = checkpoint_folder_path.joinpath('last.ckpt')
            else:
                checkpoint_path = pathlib.Path(checkpoint)

            if checkpoint_path.exists():
                config_no_checkpoint = copy.deepcopy(config)
                config_no_checkpoint.pop('checkpoint', None)
                model = MaskBevModule.load_from_checkpoint(checkpoint_path=str(checkpoint_path), strict=False,
                                                           **config_no_checkpoint)
            else:
                raise ValueError(f'Could not load checkpoint at path {checkpoint}')
        else:
            model = MaskBevModule(**config)
        return model

    def configure_optimizers(self):
        if self._differential_lr:
            grouped_parameters = [
                {'params': self._encoder.parameters(), 'lr': self._lr * self._differential_lr_scaling},
                {'params': self._backbone.parameters(), 'lr': self._lr * self._differential_lr_scaling},
                {'params': self._panoptic_head.parameters(), 'lr': self._lr},
            ]
        else:
            grouped_parameters = self.parameters()

        if self._optimiser_type == OptimizerType.ADAM:
            optimizer = torch.optim.Adam(grouped_parameters, lr=self._lr, weight_decay=self._weight_decay)
        elif self._optimiser_type == OptimizerType.LAMB:
            optimizer = torch_optimizer.Lamb(grouped_parameters, lr=self._lr, weight_decay=self._weight_decay)
        elif self._optimiser_type == OptimizerType.SGD:
            optimizer = SGD(self.parameters(), lr=self._lr, momentum=0.99, weight_decay=self._weight_decay,
                            nesterov=True)
        elif self._optimiser_type == OptimizerType.ADAM_W:
            optimizer = AdamW(grouped_parameters, lr=self._lr, weight_decay=self._weight_decay, amsgrad=False)
        else:
            raise NotImplementedError()

        # TODO poly lr schedule (1 - iter/max_iter)^0.9
        if self._lr_schedulers_type == LrSchedulerType.REDUCE_ON_PLATEAU:
            lr_scheduler = ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        elif self.lr_scheduler_type == LrSchedulerType.COSINE:
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=10)
        else:
            raise NotImplementedError()

        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            monitor='train_loss',
            interval='epoch'
        )

    def masks_to_instance_map(self, masks):
        img = torch.zeros(masks.shape[-2:], device=masks.device).unsqueeze(0)
        for i in range(masks.shape[0]):
            img += i * masks[i]
        return img

    def forward(self, x):
        x = self._encoder(x)
        x = self._backbone(x)
        x = self._panoptic_head(x)
        return x

    def forward_encode(self, pc):
        return self._encoder(pc)

    def forward_backbone(self, encoded):
        return self._backbone(encoded)

    def pred_masks(self, features):
        return self._panoptic_head(features)

    def compute_loss(self, cls, masks, labels_gt, masks_gt, heights_pred, heights_gt) -> Dict[str, Any]:
        loss_dict = self._panoptic_head.loss(cls, masks, labels_gt, masks_gt, heights_pred, heights_gt)
        return loss_dict

    def loss(self, loss_dict):
        loss = sum(value for key, value in loss_dict.items() if 'loss' in key)
        return loss

    def log_losses(self, batch_size, loss_dict, mode):
        for k, v in loss_dict.items():
            self.log(f'{mode}_{k}', v, batch_size=batch_size)
        loss_dice = sum(value for key, value in loss_dict.items() if 'dice' in key)
        loss_mask = sum(value for key, value in loss_dict.items() if 'mask' in key)
        loss_cls = sum(value for key, value in loss_dict.items() if 'cls' in key)
        loss_height = sum(value for key, value in loss_dict.items() if 'height' in key)
        self.log(f'hp_{mode}_dice', loss_dice, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'hp_{mode}_mask', loss_mask, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'hp_{mode}_cls', loss_cls, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'hp_{mode}_height', loss_height, on_step=False, on_epoch=True, batch_size=batch_size)

    def log_normalize_img(self, img):
        img = torch.log(torch.linalg.norm(img, dim=0).unsqueeze(0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        return img

    def normalize_img(self, img):
        img = torch.clip(img, -100, 100)
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        return img

    def sigmoid_img(self, img):
        img = torch.sigmoid(img)
        return img

    def training_step(self, train_batch, batch_idx):
        logging_step = batch_idx == 0
        step = self.current_epoch
        tensorboard: SummaryWriter = self.logger.experiment

        if len(train_batch) == 2:
            x, (labels_gt, masks_gt) = train_batch
            metadata = None
        elif len(train_batch) == 3:
            x, (labels_gt, masks_gt), metadata = train_batch
        else:
            raise RuntimeError('Invalid batch')
        batch_size = len(x)

        x = self._encoder(x)
        if logging_step:
            encoded_img = self.log_normalize_img(x[0])
            tensorboard.add_image('train_encoded', encoded_img, global_step=step)

        x = self._backbone(x)
        if logging_step:
            backbone_img = self.log_normalize_img(x[0][0])
            tensorboard.add_image('train_backbone', backbone_img, global_step=step)

        cls, masks, heights = self._panoptic_head(x)

        heights_gt = [m['height'] for m in metadata] if metadata is not None else [None for _ in range(len(cls))]
        loss_dict = self.compute_loss(cls, masks, labels_gt, masks_gt, heights, heights_gt)
        loss = self.loss(loss_dict)

        self._panoptic_head.add_average_precision(self._train_cls_map, self._train_height_map, self._train_mask_map,
                                                  cls,
                                                  masks, labels_gt, masks_gt, heights, heights_gt, metadata,
                                                  [self._train_easy_map, self._train_moderate_map,
                                                   self._train_hard_map], self._train_mIoU, self._train_mask_map_all)

        self.log('train_loss', loss, batch_size=batch_size, prog_bar=True)
        self.log('hp_metric', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log_losses(batch_size, loss_dict, 'train')

        if logging_step:
            instances_gt = self.masks_to_instance_map(masks_gt[0])
            tensorboard.add_image('train_gt', instances_gt, global_step=step)

            current_mask = 0
            mask_num = -1
            for i in range(masks[mask_num][0].shape[0]):
                c = cls[mask_num][0][i].argmax()
                if c > 0:
                    img = self.normalize_img(masks[mask_num][0][i]).unsqueeze(0)
                    tensorboard.add_image(f'train_pred_{current_mask}', img, global_step=step)
                    img = self.sigmoid_img(masks[mask_num][0][i]).unsqueeze(0)
                    tensorboard.add_image(f'train_pred_{current_mask}_sig', img, global_step=step)
                    current_mask += 1

        return loss

    def on_train_epoch_end(self):
        self.log('train_mIoU', self._train_mIoU.compute(), prog_bar=True)
        self.log('train_mAP_mask', self._train_mask_map.compute(), prog_bar=True)
        self.log('train_mAP_easy', self._train_easy_map.compute(), prog_bar=False)
        self.log('train_mAP_moderate', self._train_moderate_map.compute(), prog_bar=False)
        self.log('train_mAP_hard', self._train_hard_map.compute(), prog_bar=False)
        self.log('train_mAP_cls', self._train_cls_map.compute(), prog_bar=True)
        if self._train_height_map is not None:
            self.log('train_mAP_height', self._train_height_map.compute(), prog_bar=True)
        self._train_mask_map.reset()
        self._train_cls_map.reset()

    def validation_step(self, val_batch, batch_idx):
        logging_step = batch_idx == 0
        step = self.current_epoch
        tensorboard: SummaryWriter = self.logger.experiment

        if len(val_batch) == 2:
            x, (labels_gt, masks_gt) = val_batch
            metadata = None
        elif len(val_batch) == 3:
            x, (labels_gt, masks_gt), metadata = val_batch
        else:
            raise RuntimeError('Invalid batch')
        batch_size = len(x)

        x = self._encoder(x)
        if logging_step:
            encoded_img = self.log_normalize_img(x[0])
            tensorboard.add_image('val_encoded', encoded_img, global_step=step)

        x = self._backbone(x)
        if logging_step:
            backbone_img = self.log_normalize_img(x[0][0])
            tensorboard.add_image('val_backbone', backbone_img, global_step=step)

        cls, masks, heights = self._panoptic_head(x)

        # heights_gt = [m['height'] for m in metadata] if metadata is not None else [None for _ in range(len(cls))]
        heights_gt = [None for _ in range(len(cls))]
        loss_dict = self.compute_loss(cls, masks, labels_gt, masks_gt, heights, heights_gt)
        loss = self.loss(loss_dict)

        self._panoptic_head.add_average_precision(self._val_cls_map, self._val_height_map, self._val_mask_map, cls,
                                                  masks,
                                                  labels_gt, masks_gt, heights, heights_gt, metadata,
                                                  [self._val_easy_map, self._val_moderate_map, self._val_hard_map],
                                                  self._val_mIoU, self._val_mask_map_all, self._val_mask_area,
                                                  self._val_mIoU_points)

        self.log('val_loss', loss, batch_size=batch_size, prog_bar=True)
        self.log('hp_val_metric', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log_losses(batch_size, loss_dict, 'val')

        if logging_step:
            instances_gt = self.masks_to_instance_map(masks_gt[0])
            tensorboard.add_image('val_gt', instances_gt, global_step=step)
            current_mask = 0
            for i in range(masks[-1][0].shape[0]):
                c = cls[-1][0][i].argmax()
                if c > 0:
                    img = self.normalize_img(masks[-1][0][i]).unsqueeze(0)
                    tensorboard.add_image(f'val_pred_{current_mask}', img, global_step=step)
                    img = self.sigmoid_img(masks[-1][0][i]).unsqueeze(0)
                    tensorboard.add_image(f'val_pred_{current_mask}_sig', img, global_step=step)
                    current_mask += 1
        return loss

    def on_validation_end(self):
        self.log('val_mIoU', self._val_mIoU.compute(), prog_bar=True)
        self.log('val_mIoU_points', self._val_mIoU_points.compute(), prog_bar=True)

        mAP = [m.compute() for m in self._val_mask_map_all]
        self.log('val_mAP_all', sum(mAP) / 10.0, prog_bar=True)
        self.log('val_mAP_mask', self._val_mask_map.compute(), prog_bar=True)
        self.log('val_mAP_easy', self._val_easy_map.compute(), prog_bar=False)
        self.log('val_mAP_moderate', self._val_moderate_map.compute(), prog_bar=False)
        self.log('val_mAP_hard', self._val_hard_map.compute(), prog_bar=False)
        self.log('val_mAP_cls', self._val_cls_map.compute(), prog_bar=True)
        self._val_mask_area.compute()
        if self._val_height_map is not None:
            self.log('val_mAP_height', self._val_height_map.compute(), prog_bar=True)
        self._val_mask_map.reset()
        self._val_cls_map.reset()
