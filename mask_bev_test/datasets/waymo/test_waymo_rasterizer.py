import unittest

import numpy as np
from matplotlib import pyplot as plt
from torch_waymo import WaymoDataset
from torch_waymo.protocol.dataset_proto import LaserName
from torch_waymo.protocol.label_proto import Type, Label, Box

from mask_bev.datasets.waymo.waymo_rasterizer import WaymoRasterizer
from mask_bev.visualization.point_cloud_viz import show_point_cloud


class TestWaymoRasterizer(unittest.TestCase):
    def setUp(self):
        self.dataset = WaymoDataset('~/Datasets/Waymo/converted', 'training')
        self.frame = self.dataset[350]
        self.rasterizer = WaymoRasterizer((-40, 40), (-40, 40), (-20, 20), 0.16, True, 1)

    def test_generate_masks(self):
        masks = self.rasterizer.get_mask(self.frame)

        self.assertTrue(isinstance(masks, dict))
        keys = {Type.TYPE_VEHICLE}
        self.assertEquals(keys, set(masks.keys()))
        for k in keys:
            self.assertEqual((500, 500), masks[k].shape)

    @unittest.skip("graph")
    def test_masks_same_orientation(self):
        masks = self.rasterizer.get_mask(self.frame)

        plt.title('vehicles')
        plt.imshow(masks[Type.TYPE_VEHICLE])
        plt.show()
        plt.title('pedestrian')
        plt.imshow(masks[Type.TYPE_PEDESTRIAN])
        plt.show()
        plt.title('cyclist')
        plt.imshow(masks[Type.TYPE_CYCLIST])
        plt.show()
        pc = self.frame.points[LaserName.TOP.to_idx()]

        labels: [Label] = self.frame.laser_labels
        labels = list(filter(lambda label: label.type == Type.TYPE_VEHICLE, labels))
        boxes: [Box] = [label.box for label in labels]
        box_labels = np.stack(
            [[b.center_x, b.center_y, b.center_z, b.length, b.width, b.height, b.heading] for b in boxes])

        show_point_cloud('render', pc[::], box_labels=box_labels)
