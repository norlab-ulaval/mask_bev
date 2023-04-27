import unittest

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt

from mask_bev.datasets.collate_type import CollateType
from mask_bev.datasets.waymo.waymo_data_module import WaymoDataModule


class TestWaymoDataModule(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(42)
        self.batch_size = 2

    def test_tensor_batch(self):
        datamodule = WaymoDataModule('~/Datasets/Waymo/converted', self.batch_size, 1, 50, (-40, 40), (-40, 40), (-20, 20),
                                     0.16, True, collate_fn=CollateType.TensorCollate, shuffle_train=False)
        dataloader = datamodule.train_dataloader()

        (point_clouds, point_clouds_length), (labels, masks) = next(iter(dataloader))

        # (B, Mb, 4)
        max_points = max(point_clouds_length)
        self.assertEqual((self.batch_size, max_points, 3), point_clouds.shape)
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
        datamodule = WaymoDataModule('~/Datasets/Waymo/converted', self.batch_size, 1, 50, (-40, 40), (-40, 40), (-20, 20),
                                     0.16, True, collate_fn=CollateType.TensorCollate, shuffle_train=False,
                                     num_workers=0)
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

        datamodule = WaymoDataModule('~/Datasets/Waymo/converted', self.batch_size, 1, 50, (-40, 40), (-40, 40), (-20, 20),
                                     0.16, True, collate_fn=CollateType.TensorCollate, shuffle_train=True,
                                     num_workers=0)
        # Each mask should be different
        # dataloader = datamodule.val_dataloader()
        dataloader = datamodule.train_dataloader()

        point_clouds, (labels, masks) = next(iter(dataloader))
        plt.imshow(torch.flip(masks[0].sum(dim=0), dims=(0,)))
        plt.show()

        # pc = point_clouds[0][0]
        # show_point_cloud('render', pc[::])

    @unittest.skip("num queries is very long")
    def test_list_batch_num_queries_train(self):
        datamodule = WaymoDataModule('~/Datasets/Waymo/converted', self.batch_size, 1, 50, (-40, 40), (-40, 40), (-20, 20),
                                     0.16, True, collate_fn=CollateType.TensorCollate, shuffle_train=False,
                                     num_workers=0)
        dataloader = datamodule.train_dataloader()
        for pc, (labels, masks) in dataloader:
            self.assertEqual((self.batch_size, datamodule.num_queries), labels.shape)
            self.assertEqual((self.batch_size, datamodule.num_queries, 160, 160), masks.shape)
