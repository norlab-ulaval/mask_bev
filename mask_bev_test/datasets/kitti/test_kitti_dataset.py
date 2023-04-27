import unittest

import tqdm

from mask_bev.datasets.kitti.kitti_dataset import KittiDataset


class TestKittiDataset(unittest.TestCase):
    @unittest.skip('long test')
    def test_max_queries(self):
        dataset = KittiDataset('~/Datasets/KITTI', 'training')
        max_objs = 0
        for i in tqdm.tqdm(range(len(dataset))):
            frame = dataset[i]
            max_objs = max(len(frame.labels), max_objs)
        self.assertEqual(22, max_objs)
