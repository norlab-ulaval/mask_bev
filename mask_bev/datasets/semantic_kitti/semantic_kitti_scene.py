import collections
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset, SemanticKittiScan, \
    SemanticKittiCalib


@dataclass
class SemanticKittiScene:
    seq_number: int
    scan_numbers: [int]
    point_cloud: np.ndarray
    sem_label: Optional[np.ndarray]
    inst_label: Optional[np.ndarray]
    calib: SemanticKittiCalib


class SceneMaker:
    def __init__(self, max_points: int):
        """
        Build a scene from a sequence of SemanticKitti scans
        :param max_points: Maximum number of points in the sequence (usually `sequence.total_num_points()`)
        """
        self._max_points = max_points
        self._num_points = 0
        self._seq_number = None
        self._scan_numbers = []
        self._point_cloud = np.zeros((max_points, 4))
        self._sem_label = np.zeros((max_points,), dtype=np.uint32)
        self._inst_label = np.zeros((max_points,), dtype=np.uint32)
        self._calib = None
        self._has_labels = False

    def add_scan(self, scan: SemanticKittiScan):
        """
        Add a scan to the current scene
        :param scan: scan to add
        :return: None
        """
        if self._seq_number is None:
            self._seq_number = scan.seq_number
            self._calib = scan.calib
            self._has_labels = scan.has_labels
        if scan.seq_number != self._seq_number:
            raise ValueError('Scan not from same sequence')

        self._scan_numbers.append(scan.scan_number)

        pc_homo = np.copy(scan.point_cloud)
        pc_homo[:, 3] = 1

        tr_mat = scan.velo_to_pose
        pc_homo = (tr_mat @ pc_homo.T).T
        pc_homo[:, :3] /= pc_homo[:, 3].reshape((-1, 1))

        pc = np.hstack([pc_homo[:, :3], scan.point_cloud[:, 3].reshape((-1, 1))])

        num_pts = pc.shape[0]
        start_idx = self._num_points
        end_idx = start_idx + num_pts
        self._point_cloud[start_idx:end_idx, :] = pc
        if self._has_labels:
            self._sem_label[start_idx:end_idx] = scan.sem_label
            self._inst_label[start_idx:end_idx] = scan.inst_label
        self._num_points += scan.num_points

    @property
    def scene(self) -> SemanticKittiScene:
        if len(self._scan_numbers) == 0:
            raise RuntimeError('No scan in scene')
        sem_label = self._sem_label if self._has_labels else None
        inst_label = self._inst_label if self._has_labels else None
        return SemanticKittiScene(self._seq_number, self._scan_numbers, self._point_cloud, sem_label, inst_label,
                                  self._calib)


if __name__ == '__main__':
    from mask_bev.visualization.point_cloud_viz import show_point_cloud

    root = 'data/SemanticKITTI'
    train_dataset = SemanticKittiSequenceDataset(root, 'train')

    sequence = train_dataset[0]

    scan_idx = sequence.scan_indices[::10]
    max_points = sum(s.num_points for s in train_dataset.load_scan_indices(scan_idx))
    scene_maker = SceneMaker(max_points=max_points)
    for scan in tqdm.tqdm(train_dataset.load_scan_indices(scan_idx), total=len(scan_idx)):
        scene_maker.add_scan(scan)

    scan_positions = sequence.positions()
    plt.plot(scan_positions[:, 0], scan_positions[:, 1])
    plt.show()

    scene = scene_maker.scene
    pc = scene.point_cloud[::10]
    sem = scene.sem_label[::10]

    color_map = collections.defaultdict(lambda: [0, 0, 0])
    for i in range(1000):
        color_map[i] = [255, 255, 255]
    color_map[1] = [0, 0, 255]
    x, y, z, _ = train_dataset.dataset[120].pose[:, -1]

    show_point_cloud('Whole SemanticKittiScene', pc, sem, color_map=color_map)

    # poses = sequence.poses
    # t = np.linspace(0, 1, num=poses.shape[0])
    # tx = poses[:, 0, 3]
    # ty = poses[:, 1, 3]
    # tz = poses[:, 2, 3]
    # plt.scatter(tx, tz, s=t[::-1] * 50, c=t)
    # plt.show()
