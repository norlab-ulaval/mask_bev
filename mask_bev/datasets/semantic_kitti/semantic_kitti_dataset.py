import io
import pathlib
from dataclasses import dataclass
from typing import Optional, Callable, List, Union, Dict, Iterator

import numpy as np
import yaml
from torch.utils.data import Dataset


@dataclass
class SemanticKittiCalib:
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray
    velo_to_cam: np.ndarray


@dataclass
class SemanticKittiScan:
    """
    seq_number: number of the sequence the scan is contained in
    scan_number: number of the scan in the sequence
    scan: point cloud (N, 4), N=num points, 4=xyzi
    pose: homogenous matrix (4, 4)
    sem_label: semantic label (N, 1)
    inst_label: instance label (N, 1)
    """
    seq_number: int
    seq_idx: int
    scan_number: int
    point_cloud: np.ndarray
    pose: np.ndarray
    sem_label: Optional[np.ndarray]
    inst_label: Optional[np.ndarray]
    time: float
    calib: SemanticKittiCalib

    @property
    def has_labels(self):
        return self.sem_label is not None and self.inst_label is not None

    @property
    def num_points(self):
        return self.point_cloud.shape[0]

    @property
    def velo_to_cam(self):
        return self.calib.velo_to_cam

    @property
    def velo_to_pose(self):
        return np.linalg.inv(self.velo_to_cam) @ self.pose @ self.velo_to_cam

    @property
    def velo_to_inv_pose(self):
        # same as np.linalg.inv(self.velo_to_pose), but the following is clearer
        return np.linalg.inv(self.velo_to_cam) @ np.linalg.inv(self.pose) @ self.velo_to_cam

    @property
    def position(self):
        origin = np.array([0, 0, 0, 1]).T.T
        tr_mat = self.velo_to_pose
        pos = tr_mat @ origin
        return pos[:3] / pos[3]


def _positions_from_poses(poses: np.ndarray, velo_to_cam: np.ndarray) -> np.ndarray:
    origin = np.array([0, 0, 0, 1]).T.T
    tr_mat = np.linalg.inv(velo_to_cam) @ poses @ velo_to_cam
    pos = tr_mat @ origin
    return pos[:, :3] / pos[:, 3].reshape((-1, 1))


@dataclass
class SemanticKittiSequence:
    """
    seq_number: number of the sequence
    scans: list of all the scans in the sequence
    poses: transformation matrices for all poses (N, 4, 4)
    """
    seq_number: int
    scans: List[SemanticKittiScan]
    poses: np.ndarray
    calib: SemanticKittiCalib

    def total_num_points(self):
        return sum(p.num_points for p in self.scans)

    def positions(self) -> np.ndarray:
        return _positions_from_poses(self.poses, self.calib.velo_to_cam)


@dataclass
class SemanticKittiLazySequence:
    """
    seq_number: number of the sequence
    scans: list of all the scans' idx in the sequence
    poses: transformation matrices for all poses (N, 4, 4)
    """
    seq_number: int
    scan_indices: List[int]
    poses: np.ndarray
    calib: SemanticKittiCalib

    def total_num_points(self, dataset):
        return sum(dataset[i].num_points for i in self.scan_indices)

    def positions(self) -> np.ndarray:
        return _positions_from_poses(self.poses, self.calib.velo_to_cam)


class SemanticKittiRawLabel:
    """
    Raw raw_labels from dataset
    """
    UNLABELED = 0
    OUTLIER = 1
    CAR = 10
    BICYCLE = 11
    BUS = 13
    MOTORCYCLE = 15
    ON_RAILS = 16
    TRUCK = 18
    OTHER_VEHICLE = 20
    PERSON = 30
    BICYCLIST = 31
    MOTORCYCLIST = 32
    ROAD = 40
    PARKING = 44
    SIDEWALK = 48
    OTHER_GROUND = 49
    BUILDING = 50
    FENCE = 51
    OTHER_STRUCTURE = 52
    LANE_MARKING = 60
    VEGETATION = 70
    TRUNK = 71
    TERRAIN = 72
    POLE = 80
    TRAFFIC_SIGN = 81
    OTHER_OBJECT = 99
    MOVING_CAR = 252
    MOVING_BICYCLIST = 253
    MOVING_PERSON = 254
    MOVING_MOTORCYCLIST = 255
    MOVING_ON_RAILS = 256
    MOVING_BUS = 257
    MOVING_TRUCK = 258
    MOVING_OTHER_VEHICLE = 259

    @classmethod
    def all_label_names(cls):
        return [v for v in dir(cls) if v[:2] != '__']

    @classmethod
    def all_label_values(cls):
        return [cls.__dict__[v] for v in dir(cls) if v[:2] != '__']

    @classmethod
    def moving_label_names(cls):
        return [v for v in cls.all_label_names() if 'MOVING' in v]

    @classmethod
    def moving_label_values(cls):
        return [cls.__dict__[v] for v in cls.all_label_names() if 'MOVING' in v]


class SemanticKittiLearningLabel:
    """
    Labels used for learning, output of the remap done by learning_map
    """
    UNLABELED = 0
    CAR = 1
    BICYCLE = 2
    MOTORCYCLE = 3
    TRUCK = 4
    OTHER_VEHICLE = 5
    PERSON = 6
    BICYCLIST = 7
    MOTORCYCLIST = 8
    ROAD = 9
    PARKING = 10
    SIDEWALK = 11
    OTHER_GROUND = 12
    BUILDING = 13
    FENCE = 14
    VEGETATION = 15
    TRUNK = 16
    TERRAIN = 17
    POLE = 18
    TRAFFIC_SIGN = 19


class SemanticKittiDataset(Dataset):
    raw_labels = SemanticKittiRawLabel()
    learning_label = SemanticKittiLearningLabel()

    def __init__(self, root_path: str, split: str, excluded_labels: Optional[List[int]] = None,
                 included_labels: Optional[List[int]] = None, remove_unlabeled: bool = False,
                 transform: Optional[Callable] = None,
                 semantic_kitti_config: str = 'configs/semantic_kitti/semantic-kitti.yaml'):
        """
        SemanticKITTIDataset dataloader
        :param root_path: path to the dataset, should contain the `dataset` folder
        :param split: either 'train', 'valid' or 'test'
        :param excluded_labels: list of excluded raw_labels, will be remapped to unlabeled, should be from SemanticKittiRawLabel
        :param included_labels: list of included raw_labels, everything else will be remapped to unlabeled, should be from SemanticKittiRawLabel
        :param remove_unlabeled: whether to filter out unlabeled points or not
        :param transform: transform function applied to SemanticKittiScan
        :param semantic_kitti_config: path to the dataset config containing various info on the dataset
        """
        if excluded_labels is not None and included_labels is not None:
            raise ValueError('excluded_labels and included_labels can\'t be both not None')

        self._root_path = pathlib.Path(root_path).joinpath('dataset')
        with open(semantic_kitti_config, 'r') as f:
            self._config = yaml.safe_load(f)
        self._split = split
        self._transform = transform
        self._exclude_labels = excluded_labels
        self._include_labels = included_labels
        self._remove_unlabeled = remove_unlabeled

        self._label_to_name = self._config['labels']
        self._point_proportion = self._config['content']
        self._color_map = self._config['color_map']

        self._learning_map = self._config['learning_map']
        self._learning_map_lut = np.zeros((max(self._learning_map.keys()) + 100), dtype=np.int32)
        self._learning_map_lut[list(self._learning_map.keys())] = list(self._learning_map.values())
        if self._exclude_labels is not None:
            self._learning_map_lut[self._exclude_labels] = self.raw_labels.UNLABELED
        elif self._include_labels is not None:
            excluded = np.ones_like(self._learning_map_lut, dtype=bool)
            excluded[self._include_labels] = False
            self._learning_map_lut[excluded] = self.raw_labels.UNLABELED

        self._class_remap_inv = self._config['learning_map_inv']
        self._learning_ignore = self._config['learning_ignore']
        self._index_to_seq_number = sorted(self._config['split'][self._split])
        self._seq_number_to_index = {v: i for i, v in enumerate(self._index_to_seq_number)}

        sequences = self._root_path.joinpath('sequences').iterdir()
        self._all_seq = sorted(
            s for s in sequences if int(s.name) in self._index_to_seq_number)
        self._all_scans = [sorted(list(s.joinpath('velodyne').iterdir())) for s in self._all_seq]
        if self.has_labels():
            self._all_labels = [sorted(list(s.joinpath('labels').iterdir())) for s in self._all_seq]
        else:
            self._all_labels = None
        self._seq_len = [len(list(s.joinpath('velodyne').iterdir())) for s in self._all_seq]
        self._cum_seq_len = np.cumsum(self._seq_len)
        self._poses = [self._load_poses(s.joinpath('poses.txt')) for s in self._all_seq]
        self._times = [np.loadtxt(s.joinpath('times.txt')) for s in self._all_seq]
        self._calibs = [self._load_scan_calib(s.joinpath('calib.txt')) for s in self._all_seq]

    @property
    def color_map(self) -> Dict[int, List[int]]:
        return self._color_map

    @property
    def poses(self) -> List[np.ndarray]:
        return self._poses

    @property
    def root_path(self) -> pathlib.Path:
        return self._root_path

    @staticmethod
    def max_instance_value() -> int:
        return np.iinfo(np.uint16).max

    def __len__(self) -> int:
        return self._cum_seq_len[-1]

    def __getitem__(self, idx: int) -> SemanticKittiScan:
        seq_idx, scan_number = self._idx_to_seq_scan(idx)
        sample = self._get_scan(seq_idx, scan_number)

        if self._transform is not None:
            sample = self._transform(sample)

        return sample

    def get_in_sequence(self, sequence_number: int, scan_number: int) -> SemanticKittiScan:
        seq_idx = self._seq_number_to_index[sequence_number]
        return self._get_scan(seq_idx, scan_number)

    def has_labels(self) -> bool:
        return self._split != 'test'

    def get_sequence_scan_idx(self, seq_idx) -> List[int]:
        offset = 0
        if seq_idx > 0:
            offset = self._cum_seq_len[seq_idx - 1]
        length = self._seq_len[seq_idx]
        end = offset + length
        return list(range(offset, end))

    def get_sequence_scans(self, seq_idx) -> List[SemanticKittiScan]:
        return [self[i] for i in self.get_sequence_scan_idx(seq_idx)]

    def _get_scan(self, seq_idx, scan_number):
        scan_path = self._all_scans[seq_idx][scan_number]
        scan = self._load_scan(scan_path)

        if self.has_labels():
            label_path = self._all_labels[seq_idx][scan_number]
            sem_label, inst_label = self._load_label(label_path)
        else:
            sem_label, inst_label = None, None

        if self._remove_unlabeled:
            scan, sem_label, inst_label = self._filter_out_unlabeled(scan, sem_label, inst_label)

        seq_num = self._index_to_seq_number[seq_idx]
        pose = self._poses[seq_idx][scan_number]
        time = self._times[seq_idx][scan_number]
        calib = self._calibs[seq_idx]

        return SemanticKittiScan(seq_num, seq_idx, scan_number, scan, pose, sem_label, inst_label, time, calib)

    def len_seq(self) -> int:
        return len(self._all_seq)

    def _idx_to_seq_scan(self, idx):
        for seq_idx, cum_sum in enumerate(self._cum_seq_len):
            if idx < cum_sum:
                curr_len = self._seq_len[seq_idx]
                scan_number = curr_len - (cum_sum - idx)
                return seq_idx, scan_number
        raise IndexError(f'{idx} is out of range')

    def _load_poses(self, poses_path):
        """
        Load Nx4x4 transformation matrix from origin to scan
        Each line of the file is a 3x4 matrix, the last row (0 0 0 1) is omitted
        :param poses_path: path to the np file
        :return: numpy array (N, 4, 4), N=num points
        """
        poses_reduced = np.loadtxt(poses_path)
        n = poses_reduced.shape[0]
        poses_reduced = poses_reduced.reshape((n, 3, 4))
        full_poses = np.zeros((n, 4, 4))
        full_poses[:, :3, :] = poses_reduced
        full_poses[:, 3, 3] = 1
        return full_poses

    def _load_scan(self, scan_path):
        """
        Loads a velodyne point cloud from `scan_path`
        :param scan_path: path to the .bin file
        :return: numpy array (N, 4), N=num points, 4=xyzi
        """
        scan = np.fromfile(scan_path, dtype=np.float32)
        return scan.reshape((-1, 4))

    def _load_label(self, label_path):
        """
        Loads raw_labels for `label_path`
        :param label_path: path to the label file
        :return: semantic label (N, 1) and instance label (N, 1)
        """
        label = np.fromfile(label_path, dtype=np.uint32).reshape((-1))
        sem_label = (label & 0xFFFF).astype(np.uint32)  # semantic label in lower half
        inst_label = (label >> 16).astype(np.uint32)  # instance id in upper half

        # apply class remap from config
        sem_label = self._learning_map_lut[sem_label]
        inst_label[sem_label == SemanticKittiLearningLabel.UNLABELED] = 0

        return sem_label, inst_label

    def _load_scan_calib(self, calib_path):
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                k, v = line.split(':')
                mat = np.loadtxt(io.StringIO(v)).reshape((3, 4))
                if k == 'Tr':
                    calib['velo_to_cam'] = np.vstack((mat, [0, 0, 0, 1]))
                else:
                    calib[k.lower()] = mat

        return SemanticKittiCalib(**calib)

    def _filter_out_unlabeled(self, scan, sem_label, inst_label):
        mask = sem_label != self.learning_label.UNLABELED
        scan = scan[mask]
        sem_label = sem_label[mask]
        inst_label = inst_label[mask]
        return scan, sem_label, inst_label


class SemanticKittiSequenceDataset(Dataset):
    def __init__(self, root_path: str, split: str, excluded_labels: Optional[List[int]] = None,
                 included_labels: Optional[List[int]] = None, remove_unlabeled: bool = False,
                 transform: Optional[Callable] = None,
                 semantic_kitti_config: str = 'configs/semantic_kitti/semantic-kitti.yaml',
                 lazy: bool = True):
        self._dataset = SemanticKittiDataset(root_path, split, excluded_labels, included_labels, remove_unlabeled,
                                             transform, semantic_kitti_config)
        self._lazy = lazy

    @property
    def dataset(self) -> SemanticKittiDataset:
        return self._dataset

    @property
    def root_path(self) -> pathlib.Path:
        return self._dataset.root_path

    def __len__(self) -> int:
        return self.dataset.len_seq()

    def __getitem__(self, idx: int) -> Union[SemanticKittiLazySequence, SemanticKittiSequence]:
        if self._lazy:
            return self.getitem_lazy(idx)
        else:
            return self.getitem_eager(idx)

    def getitem_eager(self, idx: int) -> SemanticKittiSequence:
        scans = self.dataset.get_sequence_scans(idx)
        if len(scans) == 0:
            raise ValueError('Empty sequence')
        scan = scans[0]
        seq_num = scan.seq_number
        calib = scan.calib
        return SemanticKittiSequence(seq_num, scans, self.dataset.poses[idx], calib)

    def getitem_lazy(self, idx: int) -> SemanticKittiLazySequence:
        scans_idx = self.dataset.get_sequence_scan_idx(idx)
        if len(scans_idx) == 0:
            raise ValueError('Empty sequence')
        scan = self.dataset[scans_idx[0]]
        seq_num = scan.seq_number
        calib = scan.calib
        return SemanticKittiLazySequence(seq_num, scans_idx, self.dataset.poses[idx], calib)

    def load_scan_number_in_sequence(self, sequence: Union[SemanticKittiSequence, SemanticKittiLazySequence],
                                     scan_number: int):
        return self.dataset.get_in_sequence(sequence.seq_number, scan_number)

    def load_scan_numbers_in_sequence(self, sequence: Union[SemanticKittiSequence, SemanticKittiLazySequence],
                                      scan_numbers: List[int]) -> Iterator[SemanticKittiScan]:
        return (self.load_scan_number_in_sequence(sequence, scan_number) for scan_number in scan_numbers)

    def load_scan_index(self, scan_idx: int) -> SemanticKittiScan:
        return self.dataset[scan_idx]

    def load_scan_indices(self, scans_indices: List[int]) -> Iterator[SemanticKittiScan]:
        return (self.load_scan_index(scan_idx) for scan_idx in scans_indices)
