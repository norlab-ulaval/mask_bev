import tracemalloc
import unittest

import numpy as np

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiDataset, SemanticKittiScan, \
    SemanticKittiSequenceDataset, SemanticKittiSequence, SemanticKittiCalib, SemanticKittiRawLabel, \
    SemanticKittiLearningLabel, SemanticKittiLazySequence

tracemalloc.start()


class TestSemanticKittiDataset(unittest.TestCase):
    def setUp(self):
        self.root = 'data/SemanticKITTI'
        self.train_dataset = SemanticKittiDataset(self.root, 'train')
        self.valid_dataset = SemanticKittiDataset(self.root, 'valid')
        self.test_dataset = SemanticKittiDataset(self.root, 'test')

    def test_len(self):
        self.assertEqual(19130, len(self.train_dataset))
        self.assertEqual(4071, len(self.valid_dataset))
        self.assertEqual(20351, len(self.test_dataset))

    def test_get_item(self):
        sample = self.train_dataset[4541]

        self.assertTrue(isinstance(sample, SemanticKittiScan))
        self.assertEqual(1, sample.seq_number)
        self.assertEqual(0, sample.scan_number)
        self.assertTrue(isinstance(sample.point_cloud, np.ndarray))
        self.assertEqual(4, sample.point_cloud.shape[1])
        self.assertTrue(isinstance(sample.sem_label, np.ndarray))
        self.assertEqual(1, len(sample.sem_label.shape))
        self.assertTrue(isinstance(sample.inst_label, np.ndarray))
        self.assertEqual(1, len(sample.inst_label.shape))
        self.assertTrue(isinstance(sample.calib, SemanticKittiCalib))
        self.assertEqual((3, 4), sample.calib.p0.shape)
        self.assertEqual((4, 4), sample.calib.velo_to_cam.shape)

    def test_remap_lut(self):
        sample = self.train_dataset[4541]

        has_remapped_classes = (sample.sem_label == 252).sum() > 0
        self.assertFalse(has_remapped_classes)

    def test_poses(self):
        poses = self.train_dataset._poses[0]

        self.assertEqual(poses.shape, (4541, 4, 4))
        expected_mat = np.array([[1, 9.31323e-10, -3.27418e-11, 0],
                                 [-9.31323e-10, 1, -4.65661e-10, 7.45058e-09],
                                 [1.09139e-11, -9.31323e-10, 1, 0],
                                 [0, 0, 0, 1]])
        self.assertTrue(np.all(poses[0] == expected_mat))

    def test_scan_pose(self):
        sample = self.train_dataset[0]

        pose = sample.pose
        expected_mat = np.array([[1, 9.31323e-10, -3.27418e-11, 0],
                                 [-9.31323e-10, 1, -4.65661e-10, 7.45058e-09],
                                 [1.09139e-11, -9.31323e-10, 1, 0],
                                 [0, 0, 0, 1]])
        self.assertTrue(np.all(pose == expected_mat))

    def test_exclude(self):
        excluded_labels = [SemanticKittiRawLabel.CAR, SemanticKittiRawLabel.MOVING_CAR]
        dataset = SemanticKittiDataset(self.root, 'train', excluded_labels=excluded_labels)
        scan = dataset[0]

        has_cars = np.any(scan.sem_label == SemanticKittiLearningLabel.CAR)
        self.assertFalse(has_cars)

    def test_exclude_remove_instance_of_excluded(self):
        dataset = SemanticKittiDataset(self.root, 'train')
        sample_with_excluded = dataset[0]

        excluded_labels = [SemanticKittiRawLabel.CAR, SemanticKittiRawLabel.MOVING_CAR]
        dataset = SemanticKittiDataset(self.root, 'train', excluded_labels=excluded_labels)
        sample = dataset[0]

        labels = sample_with_excluded.sem_label
        inst = sample.inst_label
        excluded_inst = inst[labels == SemanticKittiLearningLabel.CAR]
        self.assertTrue(len(excluded_inst) > 0)
        self.assertTrue(np.all(excluded_inst == 0))

    def test_included(self):
        included_labels = [SemanticKittiRawLabel.CAR]
        dataset = SemanticKittiDataset(self.root, 'train', included_labels=included_labels)
        scan = dataset[0]

        either_cars_or_unlabeled = np.all((scan.sem_label == SemanticKittiLearningLabel.CAR) | (
                scan.sem_label == SemanticKittiLearningLabel.UNLABELED))
        self.assertTrue(either_cars_or_unlabeled)

    def test_remove_unlabeled(self):
        dataset = SemanticKittiDataset(self.root, 'train', remove_unlabeled=True)
        scan = dataset[0]

        has_unlabeled = np.any(scan.sem_label == SemanticKittiLearningLabel.UNLABELED)
        self.assertFalse(has_unlabeled)

    def test_get_in_sequence(self):
        sequence_number = 7
        scan_number = 123

        sample = self.train_dataset.get_in_sequence(sequence_number, scan_number)

        self.assertEqual(sequence_number, sample.seq_number)
        self.assertEqual(scan_number, sample.scan_number)


class TestSemanticKittiSequenceDataset(unittest.TestCase):
    def setUp(self):
        self.root = 'data/SemanticKITTI'
        self.train_dataset = SemanticKittiSequenceDataset(self.root, 'train')
        self.valid_dataset = SemanticKittiSequenceDataset(self.root, 'valid')
        self.test_dataset = SemanticKittiSequenceDataset(self.root, 'test')

    def test_len(self):
        self.assertEqual(10, len(self.train_dataset))
        self.assertEqual(1, len(self.valid_dataset))
        self.assertEqual(11, len(self.test_dataset))

    def test_lazy_getitem(self):
        sample = self.train_dataset[0]

        self.assertTrue(isinstance(sample, SemanticKittiLazySequence))

    def test_eager_getitem(self):
        dataset = SemanticKittiSequenceDataset(self.root, 'train', lazy=False)

        sample = dataset[0]
        last_sample = dataset[9]

        self.assertTrue(isinstance(sample, SemanticKittiSequence))
        self.assertEqual(4541, len(sample.scans))
        self.assertEqual(1201, len(last_sample.scans))

    def test_sequence_position(self):
        sequence = self.train_dataset[0]

        positions = sequence.positions()

        check_step = 10
        for i, scan_i in enumerate(sequence.scan_indices[::check_step]):
            scan = self.train_dataset.load_scan_index(scan_i)
            self.assertTrue(np.all(np.isclose(scan.position, positions[i * check_step])))
