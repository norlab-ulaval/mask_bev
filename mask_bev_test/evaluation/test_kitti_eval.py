import unittest

from mask_bev.datasets.kitti.kitti_dataset import KittiLabel, KittiType
from mask_bev.evaluation.kitti_eval import eval_kitti, Prediction


class TestKittiEval(unittest.TestCase):
    def test_kitti_eval(self):
        labels = [[
            KittiLabel(type=KittiType.Car, truncated=0.0, occluded=0, alpha=0.0, bbox=[0, 100, 10, 150],
                       dimensions=[10, 10, 10], location=[10, 10, 10], rotation_y=0.0) for _ in range(10)
        ] for _ in range(150)]
        predictions = [[
            Prediction(type=KittiType.Car, alpha=0.0, dimensions=[10, 100, 10], location=[10, 10, 10], rotation_y=0.0,
                       score=1.0) for _ in range(10)
        ] for _ in range(150)]
        output = eval_kitti(labels, predictions)
        print()
        print(output)
        self.assertTrue(False)
