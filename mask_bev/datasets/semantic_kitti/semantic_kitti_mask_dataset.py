import pathlib
import shutil
from dataclasses import dataclass
from typing import Callable

import numpy as np
import tqdm
from torch.utils.data import Dataset

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset, \
    SemanticKittiRawLabel, SemanticKittiScan
from mask_bev.datasets.semantic_kitti.semantic_kitti_rasterizer import SemanticKittiRasterizer
from mask_bev.datasets.semantic_kitti.semantic_kitti_scene import SceneMaker


@dataclass
class SemanticKittiMaskScan:
    scan: SemanticKittiScan
    mask: np.ndarray


class SemanticKittiMaskDataset(Dataset):
    def __init__(self, sequence_dataset: SemanticKittiSequenceDataset, x_range: (int, int), y_range: (int, int),
                 z_range: (int, int), voxel_size: float, remove_unseen: bool, min_points: int, use_cache: bool = True,
                 approx_scene: bool = False, cache_name: str = 'masks_cache', transform: Callable = None):
        """
        Dataset that generates a mask around scans
        :param sequence_dataset: base dataset
        :param x_range: range of the mask in x
        :param y_range: range of the mask in y
        :param z_range: range of the mask in z
        :param voxel_size: size of the voxels in x and y
        :param remove_unseen: remove points of objects that are not seen in the current scan
        :param min_points: minimum number of points to be considered seen
        :param transform: transform to apply on `SemanticKittiMaskScan`
        """
        self._sequence_dataset = sequence_dataset
        self._scan_dataset = self._sequence_dataset.dataset
        self._x_range = x_range
        self._y_range = y_range
        self._rasterizer = SemanticKittiRasterizer(x_range, y_range, z_range, voxel_size, remove_unseen, min_points)
        self._use_cache = use_cache
        self._approx_scene = approx_scene
        self._transform = transform

        self._cache_path = self._sequence_dataset.root_path.joinpath(cache_name)
        self._cache_hit = 0
        self._cache_miss = 0

    def clear_cache(self):
        if self._cache_path.exists():
            shutil.rmtree(str(self._cache_path))

    @property
    def cache_hit_ratio(self):
        return self._cache_hit / (self._cache_hit + self._cache_miss)

    def __len__(self) -> int:
        return len(self._scan_dataset)

    def __getitem__(self, idx: int) -> SemanticKittiMaskScan:
        scan = self._scan_dataset[idx]

        mask_scan = None
        if self._use_cache:
            cached_mask = self._get_cached(scan)
            if cached_mask is not None:
                self._cache_hit += 1
                mask_scan = SemanticKittiMaskScan(scan, cached_mask)
        if mask_scan is None:
            mask_scan = self._generate_mask(scan)

        if self._transform is not None:
            mask_scan = self._transform(mask_scan)

        return mask_scan

    def _generate_mask(self, scan):
        seq_idx = scan.seq_idx
        sequence = self._sequence_dataset[seq_idx]
        scan_positions_in_seq = sequence.positions()
        scan_positions_in_seq = np.hstack((scan_positions_in_seq, np.ones((scan_positions_in_seq.shape[0], 1))))
        scan_positions_in_seq = (scan.velo_to_inv_pose @ scan_positions_in_seq.T).T
        if self._approx_scene:
            valid_scans_numbers = self._approx_valid_scans(scan, scan_positions_in_seq)
        else:
            in_range = (self._x_range[0] < scan_positions_in_seq[:, 0]) & \
                       (scan_positions_in_seq[:, 0] < self._x_range[1]) & \
                       (self._y_range[0] < scan_positions_in_seq[:, 1]) & \
                       (scan_positions_in_seq[:, 1] < self._y_range[1])
            valid_scans_numbers = np.argwhere(in_range).squeeze()
        max_points = sum(
            s.num_points for s in self._sequence_dataset.load_scan_numbers_in_sequence(sequence, valid_scans_numbers))
        scene_maker = SceneMaker(max_points)
        for s in self._sequence_dataset.load_scan_numbers_in_sequence(sequence, valid_scans_numbers):
            scene_maker.add_scan(s)
        mask = self._rasterizer.get_mask_around(scan, scene_maker.scene)
        self._cache_mask(mask, scan)
        self._cache_miss += 1
        return SemanticKittiMaskScan(scan, mask)

    def _approx_valid_scans(self, scan, scan_positions_in_seq):
        valid_scans_numbers = []
        starting_idx = scan.scan_number
        curr_idx = starting_idx
        while curr_idx >= 0:
            in_range = (self._x_range[0] < scan_positions_in_seq[curr_idx, 0]) & \
                       (scan_positions_in_seq[curr_idx, 0] < self._x_range[1]) & \
                       (self._y_range[0] < scan_positions_in_seq[curr_idx, 1]) & \
                       (scan_positions_in_seq[curr_idx, 1] < self._y_range[1])
            if in_range:
                valid_scans_numbers.append(curr_idx)
            else:
                break
            curr_idx -= 1
        curr_idx = starting_idx
        while curr_idx < scan_positions_in_seq.shape[0]:
            in_range = (self._x_range[0] < scan_positions_in_seq[curr_idx, 0]) & \
                       (scan_positions_in_seq[curr_idx, 0] < self._x_range[1]) & \
                       (self._y_range[0] < scan_positions_in_seq[curr_idx, 1]) & \
                       (scan_positions_in_seq[curr_idx, 1] < self._y_range[1])
            if in_range:
                valid_scans_numbers.append(curr_idx)
            else:
                break
            curr_idx += 1
        return valid_scans_numbers

    def _get_cached(self, scan):
        mask_path = self._cache_of_scan(scan)
        if mask_path.exists():
            with open(str(mask_path), 'rb') as f:
                mask = np.load(f)
            return mask
        return None

    def _cache_mask(self, mask, scan):
        mask_path = self._cache_of_scan(scan)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(mask_path), 'wb') as f:
            np.save(f, mask)

    def _cache_of_scan(self, scan) -> pathlib.Path:
        seq_number = scan.seq_number
        scan_number = scan.scan_number
        return self._cache_path.joinpath(str(seq_number)).joinpath(f'{scan_number}.npy')


if __name__ == '__main__':

    root = 'data/SemanticKITTI'

    train_dataset = SemanticKittiSequenceDataset(root, 'train', included_labels=SemanticKittiRawLabel.CAR, lazy=False)
    x_range, y_range, z_range, voxel_size = (-40, 40), (-40, 40), (-10, 10), 0.16
    mask_dataset = SemanticKittiMaskDataset(train_dataset, x_range, y_range, z_range, voxel_size, remove_unseen=True,
                                            min_points=1)

    num_cars = 0
    for scan in tqdm.tqdm(mask_dataset):
        num_cars += len(set(np.unique(scan.mask)) - {0})

    print(f'{num_cars=}')

    exit(0)

    sequence = train_dataset[0]
    sample = mask_dataset[100]
    plt.imshow(sample.mask[:, ::-1].T > 0)
    plt.colorbar()
    plt.show()

    show_point_cloud('Scan', sample.scan.point_cloud, sample.scan.sem_label, color_map=train_dataset.dataset.color_map)
