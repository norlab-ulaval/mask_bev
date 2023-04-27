import tracemalloc
import unittest

import numpy as np

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset
from mask_bev.datasets.semantic_kitti.semantic_kitti_scene import SceneMaker

tracemalloc.start()


class TestSemanticKittiScene(unittest.TestCase):
    def setUp(self):
        root = 'data/SemanticKITTI'
        self.dataset = SemanticKittiSequenceDataset(root, 'train')

    def test_scene(self):
        sequence = self.dataset[0]
        scan_indices = sequence.scan_indices[:5]
        scans = list(self.dataset.load_scan_indices(scan_indices))
        total_num_pts = sum(s.num_points for s in scans)
        scene_maker = SceneMaker(total_num_pts)

        for scan in scans:
            scene_maker.add_scan(scan)

        scene = scene_maker.scene
        num_pts = 0
        for scan in scans:
            same_sem = np.all(scene.sem_label[num_pts:num_pts + scan.num_points] == scan.sem_label)
            self.assertTrue(same_sem)
            same_inst = np.all(scene.inst_label[num_pts:num_pts + scan.num_points] == scan.inst_label)
            self.assertTrue(same_inst)
            num_pts += scan.num_points
