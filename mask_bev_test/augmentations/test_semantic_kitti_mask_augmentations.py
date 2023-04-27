import unittest

import pipeline as pp
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt

from mask_bev.augmentations.semantic_kitti_mask_augmentations import Flip, ShufflePoints, RandomRotate, \
    DecimatePoints, JitterPoints, \
    RandomDropPoints
from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_data_module import SemanticKittiMaskDataModule, CollateType
from mask_bev.visualization.point_cloud_viz import show_point_cloud


class TestMaskAugmentations(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(420)
        self.batch_size = 2
        self.x_range, self.y_range, self.z_range, self.voxel_size = (-40, 40), (-40, 40), (-10, 10), 0.16

    @unittest.skip('plot')
    def test_flip_x(self):
        flip_x = Flip(prob_flip_x=1, prob_flip_y=0)
        self._run_with_augmentation(flip_x)

    @unittest.skip('plot')
    def test_flip_y(self):
        flip_y = Flip(prob_flip_x=0, prob_flip_y=1)
        self._run_with_augmentation(flip_y)

    @unittest.skip('plot')
    def test_point_shuffle(self):
        point_shuffle = ShufflePoints(1)
        self._run_with_augmentation(point_shuffle)

    @unittest.skip('plot')
    def test_random_transformation(self):
        rand_trans = RandomRotate(1, 30)
        self._run_with_augmentation(rand_trans)

    @unittest.skip('plot')
    def test_decimate(self):
        decimate = DecimatePoints(1, 10)
        self._run_with_augmentation(decimate)

    @unittest.skip('plot')
    def test_jitter(self):
        jitter = JitterPoints(1, (0.1, 0.1, 0.01), 0.05, 0.1)
        self._run_with_augmentation(jitter)

    @unittest.skip('plot')
    def test_drop_points(self):
        jitter = RandomDropPoints(1, 0.8)
        self._run_with_augmentation(jitter)

    def _show(self, masks, point_clouds):
        plt.imshow(torch.flip(masks[0].sum(dim=0), dims=(0,)))
        plt.show()
        pc = point_clouds[0][0]
        show_point_cloud('render', pc[::])

    def _get_dataloader(self, transform):
        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', self.batch_size, 1, 45, self.x_range,
                                                 self.y_range, self.z_range, self.voxel_size, True, 0, False,
                                                 collate_fn=CollateType.TensorCollate, shuffle_train=False,
                                                 dataset_transform=transform)
        dataloader = datamodule.train_dataloader()
        return dataloader

    def _run_with_augmentation(self, aug):
        id_dataloader = self._get_dataloader(pp.Identity())
        dataloader = self._get_dataloader(aug)

        id_point_clouds, (id_labels, id_masks) = next(iter(id_dataloader))
        point_clouds, (labels, masks) = next(iter(dataloader))
        self._show(id_masks, id_point_clouds)
        self._show(masks, point_clouds)
