import unittest

import matplotlib.pyplot as plt
import numpy as np

import mask_bev.utils.pipeline as pp
from mask_bev.datasets.apply_transform import ApplyTransform
from mask_bev.datasets.collate_type import CollateType
from mask_bev.datasets.kitti.kitti_dataset import KittiDataset
from mask_bev.datasets.kitti.kitti_transforms import ObjectRangeFilter, FrameToPointCloud, ShufflePointCloud, \
    FrameScanToMask, FrameMasksToLabelInstanceMasks


class TestKitti(unittest.TestCase):
    def setUp(self):
        self.x_range = (0, 70.4)
        self.y_range = (-40, 40)
        self.z_range = (-20, 20)
        self.voxel_size = 0.16
        self.min_num_points = 5
        self.dataset = KittiDataset('~/Datasets/KITTI', 'training')
        self.dataset = ApplyTransform(self.dataset, self._build_transform())

    def _build_transform(self):
        return pp.Compose([
            ObjectRangeFilter(self.x_range, self.y_range),
            pp.Tupled(2),
            pp.First(pp.Compose([
                FrameToPointCloud(),
                ShufflePointCloud(),
            ])),
            pp.Second(pp.Compose([
                FrameScanToMask(self.x_range, self.y_range, self.z_range, self.voxel_size, self.min_num_points,
                                True),
                FrameMasksToLabelInstanceMasks(50),
            ]))])

    def test_load_scan(self):
        for i, (point_clouds, (labels, mask)) in enumerate(self.dataset):
            combined_mask= mask.sum(dim=0)
            combined_mask = np.where(combined_mask > 0, 1, 0)
            plt.imshow(combined_mask)
            plt.show()
            if i > 10:
                break
