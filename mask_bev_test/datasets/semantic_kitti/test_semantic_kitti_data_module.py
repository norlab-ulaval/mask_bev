import tracemalloc
import unittest

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_data_module import SemanticKittiMaskDataModule, CollateType

tracemalloc.start()


class TestSemanticKittiDataModule(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(420)
        self.batch_size = 2
        self.x_range, self.y_range, self.z_range, self.voxel_size = (-40, 40), (-40, 40), (-10, 10), 0.16

    def test_tensor_batch(self):
        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', self.batch_size, 1, 40, self.x_range,
                                                 self.y_range, self.z_range, self.voxel_size, True, 0, False,
                                                 collate_fn=CollateType.TensorCollate)
        dataloader = datamodule.train_dataloader()

        (point_clouds, point_clouds_length), (labels, masks) = next(iter(dataloader))

        # (B, Mb, 4)
        max_points = max(point_clouds_length)
        self.assertEqual((self.batch_size, max_points, 4), point_clouds.shape)
        self.assertTrue(isinstance(point_clouds, torch.Tensor))

        # Length for each batch
        self.assertEqual((self.batch_size,), point_clouds_length.shape)
        self.assertEqual(torch.int, point_clouds_length.dtype)
        self.assertTrue(isinstance(point_clouds_length, torch.Tensor))

        # (B, num_queries)
        self.assertEqual((self.batch_size, datamodule.num_queries), labels.shape)
        self.assertEqual(torch.long, labels.dtype)
        # (B, num_queries Nx, Ny)
        self.assertEqual((self.batch_size, datamodule.num_queries, 500, 500), masks.shape)
        self.assertTrue(isinstance(masks, torch.Tensor))
        self.assertTrue(isinstance(masks, torch.Tensor))

    def test_list_batch(self):
        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', self.batch_size, 1, 40, self.x_range,
                                                 self.y_range, self.z_range, self.voxel_size, True, 0, False,
                                                 collate_fn=CollateType.TensorCollate)
        dataloader = datamodule.train_dataloader()

        point_clouds, (labels, masks) = next(iter(dataloader))

        self.assertEqual(2, len(point_clouds))
        for pc in point_clouds:
            self.assertTrue(isinstance(pc, torch.Tensor))
        self.assertEqual((self.batch_size, datamodule.num_queries), labels.shape)
        self.assertEqual(torch.long, labels.dtype)
        self.assertEqual((self.batch_size, datamodule.num_queries, 500, 500), masks.shape)
        self.assertTrue(isinstance(masks, torch.Tensor))

    @unittest.skip("plots")
    def test_data_make_sense(self):
        from mask_bev.visualization.point_cloud_viz import show_point_cloud

        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', self.batch_size, 1, 40, self.x_range,
                                                 self.y_range, self.z_range, self.voxel_size, True, 0, False,
                                                 collate_fn=CollateType.TensorCollate, shuffle_train=False)
        # Each mask should be different
        # dataloader = datamodule.val_dataloader()
        dataloader = datamodule.train_dataloader()

        point_clouds, (labels, masks) = next(iter(dataloader))
        plt.imshow(torch.flip(masks[0].sum(dim=0), dims=(0,)))
        plt.show()
        pc = point_clouds[0][0]
        show_point_cloud('render', pc[::])

    @unittest.skip("num queries is very long")
    def test_list_batch_num_queries_train(self):
        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', self.batch_size, 1, 40, self.x_range,
                                                 self.y_range, self.z_range, self.voxel_size, True, 0, False,
                                                 collate_fn=CollateType.ListCollate)
        dataloader = datamodule.train_dataloader()
        for pc, (labels, masks) in dataloader:
            self.assertEqual((self.batch_size, datamodule.num_queries), labels.shape)
            self.assertEqual((self.batch_size, datamodule.num_queries, 160, 160), masks.shape)
