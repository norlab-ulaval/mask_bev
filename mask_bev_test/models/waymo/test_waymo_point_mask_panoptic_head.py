import tracemalloc
import unittest

import torch

from mask_bev.models.head.mask_bev_panoptic_head import MaskBevPanopticHead

tracemalloc.start()


class TestWaymoMaskBevPanopticHead(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        sizes = [40, 20, 10, 5]
        dims = [2 ** i * 96 for i in range(4)]
        self.num_queries = 40
        self.panoptic_head = MaskBevPanopticHead(dims, feat_channels=256, out_channels=256,
                                                 num_queries=self.num_queries, num_classes=3)
        self.backbone_out = [torch.randn((self.batch_size, dims[i], sizes[i], sizes[i])) for i in range(4)]
        self.mask_gt = torch.randn((self.batch_size, 160, 160))
        num_classes = 4
        self.labels_gt = torch.randint(low=0, high=num_classes, size=(self.batch_size, self.num_queries))
        self.masks_gt = torch.randn((self.batch_size, self.num_queries, 160, 160))

    def test_panoptic_head(self):
        cls, masks = self.panoptic_head(self.backbone_out)

        for c, m in zip(cls, masks):
            self.assertEqual((self.batch_size, self.num_queries, 4), c.shape)
            self.assertEqual((self.batch_size, self.num_queries, 40, 40), m.shape)

    def test_loss(self):
        cls, masks = self.panoptic_head(self.backbone_out)

        loss = self.panoptic_head.loss(cls, masks, self.labels_gt, self.masks_gt)

        self.assertTrue(isinstance(loss, dict))
