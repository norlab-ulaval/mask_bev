import unittest

import torch
from sklearn.metrics import average_precision_score

from mask_bev.evaluation.average_precision import average_precision, IntegrationMode, mask_iou, batched_mask_iou


class TestAveragePrecision(unittest.TestCase):
    def setUp(self):
        self.confidences = torch.tensor([0.99, 0.98, 0.95, 0.95, 0.94, 0.92, 0.89, 0.86, 0.85, 0.82, 0.81, 0.76])
        self.tp = torch.ones(12, dtype=torch.int64)
        self.tp[8] = 0

        i = torch.randperm(12)
        self.confidences = self.confidences[i]
        self.tp = self.tp[i]
        self.total_gt = 11

    def test_ap(self):
        ap = average_precision(self.confidences, self.tp, self.total_gt)
        print(f'ap custom {ap}')
        # self.assertEqual(0.9337666775306083, ap.item())

        ap = average_precision(self.confidences, self.tp, self.total_gt, method=IntegrationMode.Continuous)
        print(f'ap cuontinous {ap}')
        # self.assertEqual(0.8958333134651184, ap.item())

        ap = average_precision(self.confidences, self.tp, self.total_gt, method=IntegrationMode.Diff)
        print(f'ap diff {ap}')
        print(ap)

    def test_ap_2(self):
        confidences = torch.tensor([0, 1, 2, 3, 4, 5, 6][::-1])
        tp = torch.tensor([True, False, True, False, False, True, False])
        num_instances = 3
        print(average_precision(confidences, tp, num_instances, IntegrationMode.InterpolationPASCAL))

    def test_scikit_ap(self):
        ap = average_precision_score(self.tp, self.confidences)
        print(ap)

    def test_disjoint_iou(self):
        mask1 = torch.zeros((10, 10), dtype=torch.float)
        mask2 = torch.zeros((10, 10), dtype=torch.float)

        mask1[1:5, 1:5] = 1
        mask1[7:, 7:] = 1

        iou = mask_iou(mask1, mask2)

        self.assertEqual(0, iou)

    def test_perfect_iou(self):
        mask1 = torch.zeros((10, 10), dtype=torch.float)
        mask2 = torch.zeros((10, 10), dtype=torch.float)

        mask1[1:5, 1:5] = 1
        mask2[1:5, 1:5] = 1

        iou = mask_iou(mask1, mask2)

        self.assertEqual(1, iou)

    def test_general_iou(self):
        mask1 = torch.zeros((10, 10), dtype=torch.float)
        mask2 = torch.zeros((10, 10), dtype=torch.float)

        mask1[0:5, 0:3] = 1
        mask2[2:7, 0:3] = 1

        iou = mask_iou(mask1, mask2)

        self.assertEqual(9 / 21, iou)

    def test_batched_iou(self):
        mask1 = torch.zeros((2, 10, 10), dtype=torch.float)
        mask2 = torch.zeros((2, 10, 10), dtype=torch.float)

        mask1[0, 0:5, 0:3] = 1
        mask2[0, 2:7, 0:3] = 1

        mask1[1, 1:5, 1:5] = 1
        mask2[1, 1:5, 1:5] = 1

        ious = batched_mask_iou(mask1, mask2)

        expected = torch.tensor([9 / 21, 1])
        self.assertTrue(torch.all(torch.isclose(expected, ious)))
