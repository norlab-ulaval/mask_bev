import tracemalloc
import unittest

import torch

from mask_bev.models.backbones.mask_bev_backbone import PointMaskBackbone

tracemalloc.start()


class TestPointMaskBackbone(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 64
        x_range, y_range, z_range, voxel_size = (0, 70.4), (-40, 40), (-10, 10), 0.5
        self.num_voxel_x = int((x_range[1] - x_range[0]) // voxel_size)
        self.num_voxel_y = int((y_range[1] - y_range[0]) // voxel_size)
        img_size = (self.num_voxel_x, self.num_voxel_y)
        self.embded_dims = 96
        self.backbone = PointMaskBackbone(img_size, self.channels, self.embded_dims, patch_size=4, window_size=10,
                                          strides=(4, 2, 2, 2), use_abs_enc=True)
        self.pseudo_img = torch.randn((self.batch_size, self.channels, self.num_voxel_x, self.num_voxel_y))

    def test_backbone(self):
        out = self.backbone(self.pseudo_img)

        self.assertEqual(4, len(out))
        expected_sizes_w = [35, 18, 9, 5]
        expected_sizes_h = [40, 20, 10, 5]
        expected_dim = [2 ** i * 96 for i in range(4)]
        for i, x in enumerate(out):
            self.assertEqual((self.batch_size, expected_dim[i], expected_sizes_w[i], expected_sizes_h[i]), x.shape)
