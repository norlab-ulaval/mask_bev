import tracemalloc
import unittest

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset
from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_dataset import SemanticKittiMaskDataset, \
    SemanticKittiMaskScan

tracemalloc.start()


class TestSemanticKittiMaskDataset(unittest.TestCase):
    def setUp(self):
        self.root = 'data/SemanticKITTI'
        self.sequence_dataset = SemanticKittiSequenceDataset(self.root, 'train')
        self.mask_dataset = SemanticKittiMaskDataset(self.sequence_dataset, (-40, 40), (-40, 40), (-10, 10), 0.5, True,
                                                     1)

    def test_mask(self):
        sample = self.mask_dataset[1]

        self.assertTrue(isinstance(sample, SemanticKittiMaskScan))

    def test_not_in_same_sequence_error(self):
        _ = self.mask_dataset[7592]
