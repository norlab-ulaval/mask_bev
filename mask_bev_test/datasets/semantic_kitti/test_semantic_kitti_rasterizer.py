import collections
import functools
import tracemalloc
import unittest
from typing import Set

import matplotlib.pyplot as plt
import numpy as np

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset
from mask_bev.datasets.semantic_kitti.semantic_kitti_rasterizer import SemanticKittiRasterizer
from mask_bev.datasets.semantic_kitti.semantic_kitti_scene import SceneMaker

tracemalloc.start()


class TestSemanticKittiRasterizer(unittest.TestCase):
    def setUp(self):
        root = 'data/SemanticKITTI'
        self._dataset = SemanticKittiSequenceDataset(root, 'train')

    def test_raster(self):
        scan, scene = self._make_scene(0, 10, 5)
        rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.1)
        mask = rasterizer.get_mask_around(scan, scene)

        self.assertEqual((800, 800), mask.shape)
        self.assertFalse(np.all(mask == 0))
        present_classes = set(scene.inst_label)
        present_classes.add(0)  # no instance
        self._assert_only_instances(present_classes, mask)

    def test_remove_unseen(self):
        scan, scene = self._make_scene(2800, 2900, 90)
        rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.5, remove_unseen=True)
        mask = rasterizer.get_mask_around(scan, scene)

        present_classes_in_scan = set(scan.inst_label)
        present_classes_in_scan.add(0)  # no instance
        self._assert_only_instances(present_classes_in_scan, mask)

    def test_weird_masks(self):
        scan, scene = self._make_scene(0, 10, 5)
        rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.1)
        mask = rasterizer.get_mask_around(scan, scene)

        plt.imshow(mask > 0)
        plt.show()


    @unittest.skip("graph")
    def test_density(self):
        c = collections.Counter()
        for i in [1, 10, 100, 250, 500, 800]:
            scan, scene = self._make_scene(10, 11, 0)
            rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.16)
            mask = rasterizer.get_mask_around(scan, scene)
            idx = rasterizer._idx
            idx_tuple = []
            for i in idx:
                idx_tuple.append(tuple(i))
            c.update(idx_tuple)
        counts = list(filter(lambda x: x < 500, c.values()))
        plt.hist(counts, bins=range(min(counts), max(counts) + 10, 10))
        plt.axvline(x=64, c='red')
        plt.show()

    def _make_scene(self, start_scene, end_scene, scan_index):
        sequence = self._dataset[0]
        scan_idx = sequence.scan_indices[start_scene:end_scene]
        scans = list(self._dataset.load_scan_indices(scan_idx))
        max_points = sum(s.num_points for s in scans)
        scene_maker = SceneMaker(max_points=max_points)
        for scan in scans:
            scene_maker.add_scan(scan)
        scan = scans[scan_index]
        scene = scene_maker.scene
        return scan, scene

    def _assert_only_instances(self, instances: Set[int], mask: np.ndarray):
        all_instances_mask = []
        for c in instances:
            all_instances_mask.append(mask == c)
        all_instances = functools.reduce(np.bitwise_or, all_instances_mask)
        self.assertTrue(np.all(all_instances))
