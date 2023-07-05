import pathlib
from dataclasses import dataclass
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


@dataclass
class KittiCalib:
    P0: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    P3: np.ndarray
    R0_rect: np.ndarray
    Tr_velo_to_cam: np.ndarray
    Tr_imu_to_velo: np.ndarray


class KittiType(IntEnum):
    Car = 0
    Van = 1
    Truck = 2
    Pedestrian = 3
    Person_sitting = 4
    Cyclist = 5
    Tram = 6
    Misc = 7
    DontCare = 8

    @classmethod
    def from_string(cls, value):
        return cls.__members__[value]

    @classmethod
    def to_string(cls, value):
        rdict = {v: k for k, v in cls.__members__.items()}
        return rdict[value]


class KittiOccluded(IntEnum):
    FullyVisible = 0
    PartlyOccluded = 1
    LargelyOccluded = 2
    Unknown = 3

    @classmethod
    def from_int(cls, value):
        inv = {v: k for k, v in cls.__members__.items()}
        return inv[value]


@dataclass
class KittiLabelCamera:
    type: KittiType
    truncated: float
    occluded: KittiOccluded
    # observation angle [-pi, pi]
    alpha: float
    bbox: np.ndarray
    # [width, height, length] in meters
    dimensions: np.ndarray
    # [x, y, z]
    location: np.ndarray
    # angle around y axis in camera coords [-pi, pi]
    rotation_y: float


@dataclass
class KittiLabel:
    type: KittiType
    truncated: float
    occluded: KittiOccluded
    # observation angle [-pi, pi]
    alpha: float
    bbox: np.ndarray
    # [length, width, height] in meters
    dimensions: np.ndarray
    # [x, y, z]
    location: np.ndarray
    # angle around y axis in camera coords [-pi, pi]
    rotation_y: float


@dataclass
class KittiFrame:
    calib: KittiCalib
    labels_camera: [KittiLabelCamera]
    labels: [KittiLabel]
    points: np.ndarray


class KittiDataset(Dataset):
    def __init__(self, root_path: str, split: str):
        self._root_path = pathlib.Path(root_path).expanduser()
        self._split = split

        self._calib_path = self._root_path.joinpath('data_object_calib').joinpath(split).joinpath('calib')
        self._label_path = self._root_path.joinpath('data_object_label_2').joinpath(split).joinpath('label_2')
        self._velodyne = self._root_path.joinpath('data_object_velodyne').joinpath(split).joinpath('velodyne')

        self._calib_files = sorted(self._calib_path.iterdir())
        self._label_files = sorted(self._label_path.iterdir())
        self._velodyne_files = sorted(self._velodyne.iterdir())

        assert len(self._calib_files) == len(self._label_files)
        assert len(self._label_files) == len(self._velodyne_files)

        self._split_path = self._root_path.joinpath(self._split)

    def __len__(self) -> int:
        return len(self._velodyne_files)

    def __getitem__(self, idx: int) -> KittiFrame:
        calib = self._get_calib(idx)
        labels_camera = self._get_labels_camera(idx)
        labels = self._labels_to_velodyne(labels_camera, calib)
        velodyne = self._get_velodyne(idx)
        return KittiFrame(calib, labels_camera, labels, velodyne)

    def _get_calib(self, idx: int) -> KittiCalib:
        with open(self._calib_files[idx], 'r') as f:
            lines = f.readlines()

        P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                       ]).reshape([3, 4])
        P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                       ]).reshape([3, 4])
        P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                       ]).reshape([3, 4])
        P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                       ]).reshape([3, 4])
        P0 = self._extend_matrix(P0)
        P1 = self._extend_matrix(P1)
        P2 = self._extend_matrix(P2)
        P3 = self._extend_matrix(P3)

        R0_rect = np.array([
            float(info) for info in lines[4].split(' ')[1:10]
        ]).reshape([3, 3])
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect

        Tr_velo_to_cam = np.array([
            float(info) for info in lines[5].split(' ')[1:13]
        ]).reshape([3, 4])
        Tr_velo_to_cam = self._extend_matrix(Tr_velo_to_cam)
        Tr_imu_to_velo = np.array([
            float(info) for info in lines[6].split(' ')[1:13]
        ]).reshape([3, 4])
        Tr_imu_to_velo = self._extend_matrix(Tr_imu_to_velo)

        return KittiCalib(P0, P1, P2, P3, rect_4x4, Tr_velo_to_cam, Tr_imu_to_velo)

    def _get_labels_camera(self, idx) -> [KittiLabelCamera]:
        with open(self._label_files[idx], 'r') as f:
            lines = f.readlines()
        contents = [line.strip().split(' ') for line in lines]
        labels = []
        for content in contents:
            label_type = KittiType.from_string(content[0])
            if label_type == KittiType.DontCare:
                continue

            truncated = float(content[1])
            occluded = KittiOccluded.from_int(int(content[2]))
            alpha = float(content[3])
            bbox = np.array([[float(info) for info in content[4:8]]]).reshape(4)
            dimensions = np.array([[float(info) for info in content[8:11]]]).reshape(3)
            location = np.array([[float(info) for info in content[11:14]]]).reshape(3)
            rotation_y = float(content[14])
            label = KittiLabelCamera(label_type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y)
            labels.append(label)
        return labels

    def _get_velodyne(self, idx) -> np.ndarray:
        return np.fromfile(self._velodyne_files[idx], dtype=np.float32).reshape((-1, 4))

    def _labels_to_velodyne(self, labels_camera: [KittiLabelCamera], calib: KittiCalib):
        v2c = calib.Tr_velo_to_cam
        c2v = np.linalg.inv(v2c)
        labels = []
        for label_cam in labels_camera:
            dimensions = label_cam.dimensions[[2, 0, 1]]
            tx, ty, tz = label_cam.location
            tx, ty, tz = (c2v @ np.array([tx, ty, tz, 1]).T)[:3]
            yaw = -label_cam.rotation_y - np.pi / 2
            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # wrap to (-pi, pi)

            label = KittiLabel(label_cam.type, label_cam.truncated, label_cam.occluded, label_cam.alpha, label_cam.bbox,
                               dimensions, np.array([tx, ty, tz]), yaw)
            labels.append(label)
        return labels

    @staticmethod
    def _extend_matrix(mat):
        mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
        return mat


if __name__ == '__main__':

    root_path = pathlib.Path('~/Datasets/KITTI').expanduser()
    dataset = KittiDataset(str(root_path), 'training')

    train_split = root_path.joinpath('train.txt')
    val_split = root_path.joinpath('val.txt')

    with open(train_split, 'r') as f:
        train_idx = [int(line.strip()) for line in f.readlines()]
    with open(val_split, 'r') as f:
        val_idx = [int(line.strip()) for line in f.readlines()]

    max_car = 0
    max_x, max_y = 0, 0
    num_empty = 0
    accepted_labels = {KittiType.Car, KittiType.Truck, KittiType.Van}
    heights = []
    num_cars = 0
    for i in val_idx:
        sample = dataset[i]
        # sample = FilterLabelDifficulty()(sample)
        labels = sample.labels
        # labels = [l for l, lc in zip(labels, sample.labels_camera) if is_difficulty_valid(l, lc)]
        labels = [l for l in labels if l.type in accepted_labels]
        num_labels = len(labels)
        max_car = max(max_car, num_labels)
        sx = max([l.location[0] for l in labels]) if num_labels > 0 else 0
        sy = max([l.location[1] for l in labels]) if num_labels > 0 else 0
        max_x = max(max_x, sx)
        max_y = max(max_y, sy)
        if num_labels == 0:
            num_empty += 1
        heights.extend([l.dimensions[2] for l in labels])
        num_cars += num_labels
    print(f'{max_car=}')
    print(f'{max_x=}')
    print(f'{max_y=}')
    print(f'{num_empty=}')
    print(f'{num_empty/len(dataset)=}')
    print(f'{np.min(heights)}')
    print(f'{np.max(heights)}')
    print(f'{num_cars=}')
    plt.hist(heights)
    plt.show()
    exit(0)

    idx = 10
    sample = dataset[idx]
    pc = sample.points
    labels = sample.labels
    box_labels = np.stack(
        [[*b.location,
          *b.dimensions,
          b.rotation_y]
         for b
         in labels])

    show_point_cloud(f'KITTI sample #{idx}', pc, box_labels=box_labels, azimuth=np.pi,
                     altitude=0.01,
                     distance=0.5)
