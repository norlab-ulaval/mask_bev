import pickle
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

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

    def test_mask_to_box(self):
        img = cv2.imread('/home/william/Pictures/mask.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cnt = cnts[0]
        ((cx, cy), (w, h), angle) = cv2.minAreaRect(cnt)
        prediction = Prediction(type=KittiType.Car, alpha=0.0, dimensions=np.array([w, h, 0]),
                                location=np.array([cx, cy, 0]), rotation_y=angle, score=1)
        print((cx, cy), (w, h), angle)

    def test_eval_from_output(self):
        with open('/home/william/Datasets/KITTI/output_val_01/762.pkl', 'rb') as f:
            (cls, masks, labels_gt, masks_gt) = pickle.load(f)
        current_mask = 0
        for i in range(masks[-1][0].shape[0]):
            c = cls[-1][0][i].argmax()
            if c > 0:
                img = self.normalize_img(masks[-1][0][i]).unsqueeze(0)
                # plt.imshow(img[0].cpu().numpy())
                # plt.show()
                img = self.sigmoid_img(masks[-1][0][i]).unsqueeze(0)
                plt.imshow(img[0].cpu().numpy())
                plt.show()
                current_mask += 1

    def normalize_img(self, img):
        img = torch.clip(img, -100, 100)
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        return img

    def sigmoid_img(self, img):
        img = torch.sigmoid(img)
        return img
