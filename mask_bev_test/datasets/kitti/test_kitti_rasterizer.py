import unittest

import numpy as np
from matplotlib import pyplot as plt

from mask_bev.datasets.kitti.kitti_dataset import KittiDataset, KittiType, KittiLabel
from mask_bev.datasets.kitti.kitti_rasterizer import KittiRasterizer
from mask_bev.visualization.point_cloud_viz import show_point_cloud


class TestKittiRasterizer(unittest.TestCase):
    def setUp(self):
        self.dataset = KittiDataset('~/Datasets/KITTI', 'training')
        self.frame = self.dataset[350]
        self.rasterizer = KittiRasterizer((0, 70.4), (-40, 40), (-20, 20), 0.16, True, 1)

    def test_generate_masks(self):
        masks = self.rasterizer.get_mask(self.frame)

        self.assertTrue(isinstance(masks, dict))
        keys = {KittiType.Car}
        self.assertEquals(keys, set(masks.keys()))
        for k in keys:
            self.assertEqual((500, 440), masks[k].shape)

    @unittest.skip("graph")
    def test_masks_same_orientation(self):
        masks = self.rasterizer.get_mask(self.frame)

        plt.title('vehicles')
        plt.imshow(masks[KittiType.Car])
        plt.show()

        pc = self.frame.points
        labels: [KittiLabel] = self.frame.labels
        labels = list(filter(lambda label: label.type == KittiType.Car, labels))
        box_labels = np.stack(
            [[*b.location,
              *b.dimensions,
              b.rotation_y]
             for b
             in labels])

        show_point_cloud('render', pc[::], box_labels=box_labels)
