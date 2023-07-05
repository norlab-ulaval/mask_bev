import tracemalloc
import unittest

import torch
from torchmetrics.detection import MeanAveragePrecision

from mask_bev.evaluation.detection_metric import BinaryClassifMapMetric, DetectionMapMetric, MeanIoU
from mask_bev.models.head.mask_bev_panoptic_head import MaskBevPanopticHead

tracemalloc.start()


class TestMaskBevPanopticHead(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        sizes = [40, 20, 10, 5]
        dims = [2 ** i * 96 for i in range(4)]
        self.num_queries = 40
        self.panoptic_head = MaskBevPanopticHead(dims, feat_channels=256, out_channels=256,
                                                 num_queries=self.num_queries, num_classes=1)
        self.backbone_out = [torch.randn((self.batch_size, dims[i], sizes[i], sizes[i])) for i in range(4)]
        self.mask_gt = torch.randn((self.batch_size, 160, 160))
        num_classes = 2
        self.num_gt = 10
        self.labels_gt = torch.zeros(size=(self.batch_size, self.num_gt), dtype=torch.long)
        self.masks_gt = torch.randint(low=0, high=1, size=(self.batch_size, self.num_gt, 160, 160))
        self.heights_gt = torch.randn((self.batch_size, self.num_gt))

    def test_panoptic_head(self):
        cls, masks, heights = self.panoptic_head(self.backbone_out)

        for c, m, h in zip(cls, masks, heights):
            self.assertEqual((self.batch_size, self.num_queries, 2), c.shape)
            self.assertEqual((self.batch_size, self.num_queries, 40, 40), m.shape)
            if h is not None:
                self.assertEqual((self.batch_size, self.num_queries), h.shape)

    def test_loss(self):
        cls, masks, heights = self.panoptic_head(self.backbone_out)

        loss = self.panoptic_head.loss(cls, masks, self.labels_gt, self.masks_gt, heights, self.heights_gt)

        self.assertTrue(isinstance(loss, dict))

    def test_mAP(self):
        cls, masks, _ = self.panoptic_head(self.backbone_out)
        cls_metric = BinaryClassifMapMetric()

        max_det = self.num_queries * self.batch_size
        map_metric = MeanAveragePrecision(iou_type='segm', max_det=max_det)
        mIoU_metric = MeanIoU()

        self.panoptic_head.update_mAP_metrics(cls, masks, self.labels_gt, self.masks_gt, cls_metric, map_metric, mIoU_metric)

        mAP = map_metric.compute()
        self.assertTrue(isinstance(mAP, dict))
