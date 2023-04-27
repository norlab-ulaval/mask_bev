import torch
from torch_waymo import SimplifiedFrame
from torch_waymo.protocol.dataset_proto import LaserName
from torch_waymo.protocol.label_proto import Type

from mask_bev.datasets.waymo.waymo_rasterizer import WaymoRasterizer


class FrameToPointCloud:
    def __init__(self, laser_name: LaserName):
        self._laser_name = laser_name

    def __call__(self, f: SimplifiedFrame):
        return torch.from_numpy(f.points[self._laser_name.to_idx()])


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


class FrameScanToMask:
    def __init__(self, x_range: (int, int), y_range: (int, int), z_range: (int, int), voxel_size: float,
                 min_num_points: int, remove_unseen: bool):
        self._rasterizer = WaymoRasterizer(x_range, y_range, z_range, voxel_size, remove_unseen, min_num_points)

    def __call__(self, f: SimplifiedFrame):
        return {k: torch.from_numpy(v) for k, v in self._rasterizer.get_mask(f).items()}


class FrameMasksToLabelInstanceMasks:
    def __init__(self, num_pred: int):
        self._num_pred = num_pred

    def __call__(self, masks: dict[Type, torch.Tensor]):
        h, w = masks[Type.TYPE_VEHICLE].shape
        labels = torch.zeros((self._num_pred,), dtype=torch.long)
        out_mask = torch.zeros((self._num_pred, h, w))

        current_mask = 0
        for label_type, mask in masks.items():
            instances = set(mask.unique().numpy()) - {0}
            for inst in instances:
                labels[current_mask] = label_type
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
    def __call__(self, f: SimplifiedFrame):
        metadata = dict(
            laser_labels=f.laser_labels,
        )
        return metadata
