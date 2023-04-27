import tracemalloc
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_data_module import CollateType
from mask_bev.datasets.waymo.waymo_data_module import WaymoDataModule
from mask_bev.models.encoders.mask_bev_encoders import PointMaskEncoder, EncodingType

tracemalloc.start()


class TestWaymoPointMaskEncoders(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        z_range = (-20, 20)
        x_range, y_range, z_range, voxel_size = (-40, 40), (-40, 40), z_range, 0.16
        self.datamodule = WaymoDataModule('~/Datasets/Waymo/converted', self.batch_size, 1, 45,
                                          x_range, y_range, z_range, voxel_size, True, 0, False,
                                          collate_fn=CollateType.ListCollate, shuffle_train=True)
        dataloader = self.datamodule.train_dataloader()
        self.batch = next(iter(dataloader))

        voxel_size_z = z_range[1] - z_range[0]
        self.num_voxel_x = int((x_range[1] - x_range[0] + 1e-4) // voxel_size)
        self.num_voxel_y = int((y_range[1] - y_range[0] + 1e-4) // voxel_size)
        self.num_voxel_z = 1
        self.num_voxels = self.num_voxel_x * self.num_voxel_y
        self.max_num_points = 100
        self.feat_channels = [16, 32, 64]
        self.channels = self.feat_channels[-1]
        self.encoder = PointMaskEncoder(self.feat_channels, x_range, y_range, z_range, voxel_size,
                                        voxel_size, voxel_size_z, self.max_num_points,
                                        encoding_type=EncodingType.Vanilla, fourier_enc_group=4, pc_point_dim=3)

    def test_voxelization(self):
        x, y, _ = self.batch
        voxels, num_points, coors_batch = self.encoder.voxelize(x)

        # voxels [V, max_point, 4] -> (x, y, _, z, r)
        # the voxel at index i contains num_points[i] points, the rest is 0
        num_voxels = voxels.shape[0]
        self.assertEqual((num_voxels, self.max_num_points, 3), voxels.shape)

        # num_points [V]
        self.assertEqual((num_voxels,), num_points.shape)
        self.assertTrue(torch.all(0 <= num_points))
        self.assertTrue(torch.all(num_points <= self.max_num_points))

        # coors_batch [V, 4] -> (batch, z, y, x)
        self.assertEqual((num_voxels, 4), coors_batch.shape)
        self.assertTrue(torch.all(0 <= coors_batch[:, :]))
        self.assertTrue(torch.all(coors_batch[:, 0] <= self.batch_size))  # batch
        self.assertTrue(torch.all(coors_batch[:, 3] <= self.num_voxel_x))  # x
        self.assertTrue(torch.all(coors_batch[:, 2] <= self.num_voxel_y))  # y
        self.assertTrue(torch.all(coors_batch[:, 1] < 1))  # z

    def test_encoding(self):
        x, y, _ = self.batch
        voxels, num_points, coors_batch = self.encoder.voxelize(x)
        num_voxels = voxels.shape[0]

        voxel_features = self.encoder.encode(voxels, num_points, coors_batch)

        self.assertEqual((num_voxels, self.channels), voxel_features.shape)

    def test_forward(self):
        x, y, _ = self.batch

        features = self.encoder(x)

        self.assertEqual((self.batch_size, self.channels, self.num_voxel_x, self.num_voxel_y), features.shape)

    @unittest.skip("graph")
    def test_show_encoding(self):
        x, y, _ = self.batch

        features = self.encoder(x).detach().numpy()[0]

        a = features.reshape(64, -1)
        plt.hist(a[0, :])
        plt.show()

        img = np.linalg.norm(features, axis=0)
        img = np.log(img)
        img /= img.max()
        plt.imshow(img)
        plt.show()

        masks = y[1].detach().numpy()[0]
        mask = np.sum(masks, axis=0)
        plt.imshow(mask)
        plt.show()

        plt.imshow(mask + img)
        plt.show()

        # show_point_cloud('Whole SemanticKittiScene', x[0])
