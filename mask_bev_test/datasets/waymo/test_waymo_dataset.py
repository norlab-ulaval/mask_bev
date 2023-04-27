import unittest

import tqdm
from torch_waymo import WaymoDataset


class TestWaymoDataset(unittest.TestCase):
    @unittest.skip('long test')
    def test_max_queries(self):
        dataset = WaymoDataset('~/Datasets/Waymo/converted', 'train')
        max_objs = 0
        for i in tqdm.tqdm(range(len(dataset))):
            frame = dataset[i]
            max_objs = max(len(frame.laser_labels), max_objs)
        print(max_objs)
