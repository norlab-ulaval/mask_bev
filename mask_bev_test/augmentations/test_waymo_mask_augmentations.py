import unittest

import numpy as np
import pipeline as pp
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt

from mask_bev.augmentations.waymo_mask_augmentations import Flip, ShufflePoints, RandomRotate, DecimatePoints, \
    JitterPoints, RandomDropPoints
from mask_bev.datasets.collate_type import CollateType
from mask_bev.datasets.waymo.waymo_data_module import WaymoDataModule
from mask_bev.models.encoders.mask_bev_encoders import MaskBevEncoder, EncodingType


class TestMaskAugmentations(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.x_range, self.y_range, self.z_range, self.voxel_size = (-40, 40), (-40, 40), (-10, 10), 0.16
        self.z_range = (-20, 20)
        self.feat_channels = [16, 32, 64]
        voxel_size = 0.16
        voxel_size_z = self.z_range[1] - self.z_range[0]
        self.max_num_points = 100
        self.encoder = MaskBevEncoder(self.feat_channels, self.x_range, self.y_range, self.z_range, voxel_size,
                                      voxel_size, voxel_size_z, self.max_num_points,
                                      encoding_type=EncodingType.Vanilla, fourier_enc_group=4, pc_point_dim=3)

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
        rand_trans = RandomRotate(1, 180)
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
        features = self.encoder(point_clouds).detach().numpy()[0]
        img = np.linalg.norm(features, axis=0)
        img = np.log(img)
        img /= img.max()
        mask = torch.sum(masks[0], dim=0)
        plt.imshow(mask + img)
        plt.show()

    def _get_dataloader(self, transform):
        datamodule = WaymoDataModule('~/Datasets/Waymo/converted', self.batch_size, 1, 45,
                                     self.x_range, self.y_range, self.z_range, self.voxel_size, True, 0, False,
                                     collate_fn=CollateType.ListCollate, shuffle_train=False, frame_transform=transform)
        dataloader = datamodule.train_dataloader()
        return dataloader

    def _run_with_augmentation(self, aug):
        pl.seed_everything(42)
        id_dataloader = self._get_dataloader(pp.Identity())
        pl.seed_everything(42)
        dataloader = self._get_dataloader(aug)

        id_iter = iter(id_dataloader)
        for i in range(10): next(id_iter)
        iter_ = iter(dataloader)
        for i in range(10): next(iter_)

        id_point_clouds, (id_labels, id_masks) = next(id_iter)
        point_clouds, (labels, masks) = next(iter_)
        self._show(id_masks, id_point_clouds)
        self._show(masks, point_clouds)
