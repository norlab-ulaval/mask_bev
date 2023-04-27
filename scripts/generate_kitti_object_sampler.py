import pickle

import numpy as np
import tqdm
from mmdet3d.structures.ops import box_np_ops

from mask_bev.augmentations.kitti_mask_augmentations import label_to_array, Sample
from mask_bev.datasets.kitti.kitti_dataset import KittiDataset

MIN_PTS = 5

dataset = KittiDataset('data/KITTI', 'training')
samples = []

for frame in tqdm.tqdm(dataset):
    points = frame.points[:, :3]
    bboxes = np.stack([label_to_array(x) for x in frame.labels])
    mask = box_np_ops.points_in_rbbox(points, bboxes)
    num_labels = len(frame.labels)

    for i in range(num_labels):
        m = mask[:, i]
        if m.sum() >= MIN_PTS:
            pts = frame.points[m]
            s = Sample(pts, frame.labels[i], frame.labels_camera[i])
            samples.append(s)

with open('data/KITTI/samples.pkl', 'wb') as f:
    pickle.dump(samples, f)
