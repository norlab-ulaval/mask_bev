import pathlib
import pickle

import numpy as np
import torch

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiScan, SemanticKittiLearningLabel
from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_dataset import SemanticKittiMaskScan


class FilterSmallMasks:
    def __init__(self, min_num_inst_pixels: int):
        """
        Filter small masks under a certain number of pixels
        :param min_num_inst_pixels:
        """
        self._min_num_inst_pixels = min_num_inst_pixels

    def __call__(self, s: SemanticKittiMaskScan):
        for inst in np.unique(s.mask):
            if inst == 0:
                continue
            if np.sum(s.mask == inst) < self._min_num_inst_pixels:
                s.mask[s.mask == inst] = 0


class ScanToPointCloud:
    def __call__(self, s: SemanticKittiScan):
        return s.point_cloud


class ScanListCollate:
    def __call__(self, batch):
        return [torch.from_numpy(b) for b in batch]


class ScanTensorCollate:
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


class MaskScanToPointCloud:
    def __call__(self, s: SemanticKittiMaskScan):
        return torch.from_numpy(s.scan.point_cloud)


class ShufflePointCloud:
    def __call__(self, pc: torch.Tensor):
        idx = torch.randperm(pc.shape[0])
        return pc[idx]


class MaskScanToMask:
    def __call__(self, s: SemanticKittiMaskScan):
        return torch.from_numpy(s.mask)


class MaskToLabelInstanceMasks:
    def __init__(self, num_pred: int):
        self._num_pred = num_pred

    def __call__(self, mask: torch.Tensor):
        mask = mask.T
        h, w = mask.shape
        instances = set(mask.unique().numpy()) - {0}
        labels = torch.zeros((self._num_pred,), dtype=torch.long)
        masks = torch.zeros((self._num_pred, h, w))
        for i, inst in enumerate(instances):
            labels[i] = SemanticKittiLearningLabel.CAR
            masks[i, mask == inst] = 1.0
        return labels, masks


class LabelMaskToMask2FormerLabel:
    def __init__(self, num_classes: int):
        self._num_classes = num_classes

    def __call__(self, x: tuple[torch.Tensor, torch.Tensor]):
        labels, masks = x
        labels = self._num_classes - labels
        return labels, masks


class MaskListCollateHeight:
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


class MaskListCollate:
    def __call__(self, batch):
        """
        Collate point clouds in a list
        :param batch: batch to collate (list of tuple (x, y))
        :return: (list of point clouds, tensor of y)
        """
        pc = [b[0] for b in batch]
        labels = torch.stack([b[1][0] for b in batch])
        masks = torch.stack([b[1][1] for b in batch])
        return pc, (labels, masks)


class MaskTensorCollate:
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
    def __call__(self, f: SemanticKittiMaskScan):
        metadata = dict(
            scan=f.scan,
            mask=f.mask,
        )
        return metadata


class FrameRoundedHeight:
    def __init__(self):
        self.cache_folder = pathlib.Path('data/SemanticKITTI/heights/')
        self.cache_folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, x: dict):
        s: SemanticKittiScan = x['scan']
        name = s.seq_number
        path = self.cache_folder.joinpath(f'{name}.pkl')

        if not path.exists():
            raise RuntimeError('no height cache found')

        with open(path, 'rb') as f:
            height_map = pickle.load(f)

        heights = []
        for inst in set(np.unique(s.inst_label)) - {0}:
            heights.append(height_map[inst])
        e = 5
        clip = np.clip([round(h * e) / e for h in heights], 1, 3)
        with open(path, 'wb') as f:
            pickle.dump(clip, f)
        x['height'] = clip
        return x
