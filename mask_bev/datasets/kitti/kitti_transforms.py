from enum import IntEnum

import numpy as np
import torch

from mask_bev.datasets.kitti.kitti_dataset import KittiFrame, KittiType, KittiOccluded, KittiLabel, KittiLabelCamera
from mask_bev.datasets.kitti.kitti_rasterizer import KittiRasterizer


class FrameToPointCloud:
    def __call__(self, f: KittiFrame):
        return torch.from_numpy(f.points)


class ScanListCollate:
    def __call__(self, batch):
        return [torch.from_numpy(b) for b in batch]


class FrameTensorCollate:
    def __call__(self, batch):
        batch_size = len(batch)
        pc_dim = batch[0].shape[1]
        max_pts = max(b.shape[0] for b in batch)

        pcs = torch.zeros((batch_size, max_pts, pc_dim))
        pc_len = torch.zeros((batch_size,), dtype=torch.int)
        for i, b in enumerate(batch):
            length = b.shape[0]
            pcs[i, :length, :] = torch.from_numpy(b)
            pc_len[i] = length
        return pcs, pc_len


class ShufflePointCloud:
    def __call__(self, pc: torch.Tensor):
        idx = torch.randperm(pc.shape[0])
        return pc[idx]


class Difficulty(IntEnum):
    Easy = 1
    Moderate = 2
    Hard = 3
    Other = 4


def is_difficulty_valid(label: KittiLabel, label_camera: KittiLabelCamera):
    bbox_cam = label_camera.bbox
    bbox_height = bbox_cam[3] - bbox_cam[1]
    occ = label.occluded
    trunc = label.truncated

    if occ == KittiOccluded.FullyVisible and trunc < 0.15:
        return True
    elif occ == KittiOccluded.PartlyOccluded and trunc <= 0.3:
        return True
    elif occ == KittiOccluded.LargelyOccluded and trunc <= 0.5:
        return True

    return False


class FilterLabelDifficulty:
    def __call__(self, f: KittiFrame):
        is_valid_label = [is_difficulty_valid(label, label_camera) for label, label_camera in
                          zip(f.labels, f.labels_camera)]
        new_labels = []
        new_labels_camera = []
        for i, is_valid in enumerate(range(len(is_valid_label))):
            if is_valid:
                new_labels.append(f.labels[i])
                new_labels_camera.append(f.labels_camera[i])
        f.labels = new_labels
        f.labels_camera = new_labels_camera
        return f


class FrameScanToMask:
    def __init__(self, x_range: (int, int), y_range: (int, int), z_range: (int, int), voxel_size: float,
                 min_num_points: int, remove_unseen: bool):
        self._rasterizer = KittiRasterizer(x_range, y_range, z_range, voxel_size, remove_unseen, min_num_points)

    def __call__(self, f: KittiFrame):
        return {k: torch.from_numpy(v) for k, v in self._rasterizer.get_mask(f).items()}


class FrameMasksToLabelInstanceMasks:
    def __init__(self, num_pred: int):
        self._num_pred = num_pred

    def __call__(self, masks: dict[KittiType, torch.Tensor]):
        h, w = masks[KittiType.Car].shape
        labels = torch.zeros((self._num_pred,), dtype=torch.long)
        out_mask = torch.zeros((self._num_pred, h, w))

        current_mask = 0
        for label_type, mask in masks.items():
            instances = set(mask.unique().numpy()) - {0}
            for inst in instances:
                labels[current_mask] = label_type + 1
                out_mask[current_mask, mask == inst] = 1.0
                current_mask += 1
        return labels, out_mask


class FrameMaskListCollate:
    def __call__(self, batch):
        """
        Collate point clouds in a list
        :param batch: batch to collate (list of tuple (x, y))
        :return: (list of point clouds, tensor of y)
        """
        pc = [b[0] for b in batch]
        labels = torch.stack([b[1][0] for b in batch])
        masks = torch.stack([b[1][1] for b in batch])
        metadata = [b[2] for b in batch]
        return pc, (labels, masks), metadata


class FrameMaskTensorCollate:
    def __call__(self, batch):
        """
        Collate point clouds to a tensor, based on max number of points in batch
        :param batch: batch to collate (list of tuple (x, y))
        :return: ((point clouds tensor, list of length), tensor of y)
        """
        batch_size = len(batch)
        pc_dim = batch[0][0].shape[1]
        max_pts = max(b[0].shape[0] for b in batch)

        pcs = torch.zeros((batch_size, max_pts, pc_dim))
        pc_len = torch.zeros((batch_size,), dtype=torch.int)
        for i, b in enumerate(batch):
            length = b[0].shape[0]
            pcs[i, :length, :] = b[0]
            pc_len[i] = length
        labels = torch.stack([b[1][0] for b in batch])
        masks = torch.stack([b[1][1] for b in batch])
        return (pcs, pc_len), (labels, masks)


class FrameMetaData:
    def __call__(self, f: KittiFrame):
        metadata = dict(
            calib=f.calib,
            labels_camera=f.labels_camera,
            labels=f.labels
        )
        return metadata


class FrameDifficulty:
    def __init__(self):
        self._car_like_labels = {KittiType.Car, KittiType.Van, KittiType.Truck}

    def __call__(self, x: dict):
        label_types = [b.type for b in x['labels_camera']]
        bbox_cam = [b.bbox for b in x['labels_camera']]
        bbox_height = [b[3] - b[1] for b in bbox_cam]
        occlusion = [b.occluded for b in x['labels']]
        truncation = [b.truncated for b in x['labels']]

        difficulties = []
        for t, h, occ, trunc in zip(label_types, bbox_height, occlusion, truncation):
            if t not in self._car_like_labels:
                continue
            if occ == 'FullyVisible':
                occ = KittiOccluded.FullyVisible
            elif occ == 'PartlyOccluded':
                occ = KittiOccluded.PartlyOccluded
            elif occ == 'LargelyOccluded':
                occ = KittiOccluded.LargelyOccluded
            elif occ == 'Unknown':
                occ = KittiOccluded.Unknown

            if occ <= KittiOccluded.FullyVisible and trunc < 0.15:
                difficulties.append(Difficulty.Easy)
            elif occ <= KittiOccluded.PartlyOccluded and trunc <= 0.3:
                difficulties.append(Difficulty.Moderate)
            elif occ == KittiOccluded.LargelyOccluded and trunc <= 0.5:
                difficulties.append(Difficulty.Hard)
            else:
                difficulties.append(Difficulty.Other)
        x['difficulty'] = difficulties
        return x


class ObjectRangeFilter:
    def __init__(self, range_x, range_y):
        self._range_x = range_x
        self._range_y = range_y

    def __call__(self, f: KittiFrame):
        is_valid_label = [self._in_range(label) for label in f.labels]
        new_labels = []
        new_labels_camera = []
        for i, is_valid in enumerate(is_valid_label):
            if is_valid:
                new_labels.append(f.labels[i])
                new_labels_camera.append(f.labels_camera[i])
        f.labels = new_labels
        f.labels_camera = new_labels_camera
        return f

    def _in_range(self, label: KittiLabel):
        x = label.location[0]
        y = label.location[1]
        return self._range_x[0] <= x <= self._range_x[1] and self._range_y[0] <= y <= self._range_y[1]


class FrameRoundedHeight:
    def __call__(self, x: dict):
        e = 5
        x['height'] = np.clip([round(b.dimensions[2] * e) / e for b in x['labels']], 1, 3)
        return x
