import unittest

import numpy as np
import pipeline as pp
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt

from mask_bev.augmentations.kitti_mask_augmentations import Flip, ShufflePoints, RandomRotate, DecimatePoints, \
    JitterPoints, RandomDropPoints, GlobalNoise, BoxNoise, ObjectSample
from mask_bev.datasets.collate_type import CollateType
from mask_bev.datasets.kitti.kitti_data_module import KittiDataModule
from mask_bev.models.encoders.mask_bev_encoders import PointMaskEncoder, EncodingType


class TestMaskAugmentations(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.x_range, self.y_range, self.z_range, self.voxel_size = (0, 80), (-40, 40), (-20, 20), 0.16
        self.feat_channels = [16, 32, 64]
        voxel_size = 0.16
        voxel_size_z = self.z_range[1] - self.z_range[0]
        self.max_num_points = 100
        self.sample_idx = 1
        self.seed = 50
        self.encoder = PointMaskEncoder(self.feat_channels, self.x_range, self.y_range, self.z_range, voxel_size,
                                        voxel_size, voxel_size_z, self.max_num_points,
                                        encoding_type=EncodingType.Vanilla, fourier_enc_group=4)

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
        rand_trans = RandomRotate(1, 5)
        self._run_with_augmentation(rand_trans)

    @unittest.skip('plot')
    def test_decimate(self):
        decimate = DecimatePoints(1, 10)
        self._run_with_augmentation(decimate)

    @unittest.skip('plot')
    def test_jitter(self):
        jitter = JitterPoints(1, (0.01, 0.01, 0.01), 0.02, 0.01)
        self._run_with_augmentation(jitter)

    @unittest.skip('plot')
    def test_drop_points(self):
        drop = RandomDropPoints(1, 0.05)
        self._run_with_augmentation(drop)

    @unittest.skip('plot')
    def test_global_noise(self):
        global_noise = GlobalNoise(1, 0.2)
        self._run_with_augmentation(global_noise)

    @unittest.skip('plot')
    def test_box_noise(self):
        box_noise = BoxNoise(rot_range=np.pi / 20, translation_std=0.25, global_rot_range=0)
        self._run_with_augmentation(box_noise)

    @unittest.skip('plot')
    def test_object_sample(self):
        object_sample = ObjectSample('data/KITTI', 16)
        self._run_with_augmentation(object_sample)

    def _show(self, masks, point_clouds):
        features = self.encoder(point_clouds).detach().numpy()[0]
        img = np.linalg.norm(features, axis=0)
        img = np.log(img)
        img /= img.max()
        mask = torch.sum(masks[0], dim=0)
        plt.imshow(mask + img)
        plt.show()
        # show_point_cloud('pc', point_clouds[0])

    def _get_dataloader(self, transform):
        datamodule = KittiDataModule('data/KITTI', self.batch_size, 1, 45,
                                     self.x_range, self.y_range, self.z_range, self.voxel_size, True, 0, False,
                                     collate_fn=CollateType.ListCollate, shuffle_train=False, frame_transform=transform)
        dataloader = datamodule.train_dataloader()
        return dataloader

    def _run_with_augmentation(self, aug):
        pl.seed_everything(self.seed)
        id_dataloader = self._get_dataloader(pp.Identity())
        pl.seed_everything(self.seed)
        dataloader = self._get_dataloader(aug)

        iterator_id = iter(id_dataloader)
        iterator = iter(dataloader)
        for _ in range(self.sample_idx):
            id_point_clouds, (id_labels, id_masks), metadata_id = next(iterator_id)
            point_clouds, (labels, masks), metadata = next(iterator)
        self._show(id_masks, id_point_clouds)
        self._show(masks, point_clouds)
