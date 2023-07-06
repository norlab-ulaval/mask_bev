import copy
import numbers
import pathlib
import pickle
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, List, Callable

import numpy as np
import torch
from mmdet3d.datasets.transforms import data_augment_utils
from mmdet3d.datasets.transforms.data_augment_utils import noise_per_object_v3_
from mmdet3d.structures.ops import box_np_ops
from torchvision.transforms import RandomErasing

from mask_bev.augmentations.rand_augment import RandAugment
from mask_bev.datasets.kitti.kitti_dataset import KittiFrame, KittiLabel, KittiLabelCamera


def make_augmentation(args) -> Callable:
    name = args.get('name')
    if name == 'flip':
        constructor = Flip
    elif name == 'shuffle':
        constructor = ShufflePoints
    elif name == 'rotate':
        constructor = RandomRotate
    elif name == 'decimate':
        constructor = DecimatePoints
    elif name == 'jitter':
        constructor = JitterPoints
    elif name == 'drop':
        constructor = RandomDropPoints
    elif name == 'rand_augment':
        transforms = make_kitti_augmentation_list(args.get('transforms'))
        return RandAugment(args.get('num_augments'), transforms, args.get('magnitude'))
    elif name == 'cut_pc':
        constructor = CutPcAugmentation
    elif name == 'global_noise':
        constructor = GlobalNoise
    elif name == 'object_noise':
        constructor = BoxNoise
    elif name == 'object_sample':
        constructor = ObjectSample
    else:
        raise NotImplementedError(f'{name} is not implemented')
    kwargs = copy.copy(args)
    kwargs.pop('name')
    return constructor(**kwargs)


def make_kitti_augmentation_list(augmentations: List[Dict]) -> [Callable]:
    return [make_augmentation(aug) for aug in augmentations]


class Flip:
    def __init__(self, prob_flip_x: float = 0, prob_flip_y: float = 0.5):
        if prob_flip_x != 0:
            raise ValueError('Cannot flip in x')
        self._prob_flip_y = prob_flip_y

    def __call__(self, x: KittiFrame, magnitude: float = 1) -> KittiFrame:
        if np.random.uniform(0, 1) < self._prob_flip_y * magnitude:
            x.points[:, 1] = -x.points[:, 1]
            x.labels = [self._flip_y_label(label) for label in x.labels]
        return x

    def _flip_y_label(self, label: KittiLabel) -> KittiLabel:
        location = label.location
        location[1] = -location[1]
        return KittiLabel(label.type, label.truncated, label.occluded, -label.alpha, label.bbox, label.dimensions,
                          location, -label.rotation_y)


class ShufflePoints:
    def __init__(self, prob_shuffle: float = 0.5):
        self._prob_shuffle = prob_shuffle

    def __call__(self, x: KittiFrame, magnitude: float = 1) -> KittiFrame:
        if np.random.uniform(0, 1) < self._prob_shuffle * magnitude:
            np.random.shuffle(x.points)
        return x


class RandomRotate:
    def __init__(self, rotate_prob: float, rotation_range: Union[float, Tuple[float, float]]):
        """
        Random rotation of scan and mask
        :param rotate_prob: probability to apply a rotation
        :param rotation_range: range of possible rotation in degrees, either `angle` (-angle, angle) or `(min, max)`
        """
        self._rotate_prob = rotate_prob
        if isinstance(rotation_range, numbers.Number):
            rotation_range = (-rotation_range, rotation_range)
        self._rotation_range = rotation_range

    def __call__(self, x: KittiFrame, magnitude: float = 1) -> KittiFrame:
        if np.random.uniform(0, 1) < self._rotate_prob:
            rotation_range = (self._rotation_range[0] * magnitude, self._rotation_range[1] * magnitude)
            theta = np.random.uniform(*rotation_range)

            c = np.cos(np.deg2rad(theta))
            s = np.sin(np.deg2rad(theta))
            R = np.array([[c, -s, 0, 0],
                          [s, c, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

            scene_pc_homo = np.copy(x.points)
            scene_pc_homo[:, 3] = 1
            scene_pc_homo = (R @ scene_pc_homo.T).T
            scene_pc_homo /= scene_pc_homo[:, 3].reshape((-1, 1))
            x.points[:, :3] = scene_pc_homo[:, :3]

            x.labels = [self._rotate_label(label, theta, R) for label in x.labels]

        return x

    def _rotate_label(self, label: KittiLabel, angle: float, R: np.ndarray) -> KittiLabel:
        angle = np.deg2rad(angle)
        location = np.array([*label.location, 1])
        location = (R @ location.T).T
        return KittiLabel(label.type, label.truncated, label.occluded, label.alpha + angle, label.bbox,
                          label.dimensions, location[:3], label.rotation_y + angle)


class DecimatePoints:
    def __init__(self, prob_decimate: float, keep_every: int):
        self._prob_decimate = prob_decimate
        self._keep_every = keep_every

    def __call__(self, x: KittiFrame, magnitude: float = 1) -> KittiFrame:
        if np.random.uniform(0, 1) < self._prob_decimate:
            pc = x.points
            x.points = pc[torch.randperm(pc.shape[0])][::int(self._keep_every * magnitude)]
        return x


class JitterPoints:
    def __init__(self, prob_jitter: float, jitter_std: Union[float, Tuple[float, float, float]],
                 max_delta: Optional[Union[float, Tuple[float, float, float]]] = None, intensity_std: float = 0.0,
                 intensity_max_delta: Optional[float] = None):
        self._prob_jitter = prob_jitter
        if isinstance(jitter_std, numbers.Number):
            jitter_std = (jitter_std, jitter_std, jitter_std)
        if isinstance(max_delta, numbers.Number):
            max_delta = (max_delta, max_delta, max_delta)
        self._jitter_std = jitter_std
        self._max_delta = max_delta
        self._intensity_std = intensity_std
        self._intensity_max_delta = intensity_max_delta

    def __call__(self, x: KittiFrame, magnitude: float = 1) -> KittiFrame:
        if np.random.uniform(0, 1) < self._prob_jitter:
            noise = np.random.standard_normal(x.points.shape)
            for dim in range(3):
                noise[:, dim] *= self._jitter_std[dim]
            if self._max_delta is not None:
                for dim in range(3):
                    np.clip(noise[:, dim], -self._max_delta[dim], self._max_delta[dim], noise[:, dim])
            noise[:, 3] *= self._intensity_std
            if self._intensity_max_delta is not None:
                np.clip(noise[:, 3], -self._intensity_max_delta, self._intensity_max_delta, noise[:, 3])

            x.points += noise * magnitude
            np.clip(x.points[:, 3], 0, 1, x.points[:, 3])
        return x


class RandomDropPoints:
    def __init__(self, prob_drop: float, per_point_drop_prob: float):
        self._prob_drop = prob_drop
        self._per_point_drop_prob = per_point_drop_prob

    def __call__(self, x: KittiFrame, magnitude: float = 1) -> KittiFrame:
        if np.random.uniform(0, 1) < self._prob_drop:
            n = x.points.shape[0]
            keep = np.random.uniform(0, 1, n) >= self._per_point_drop_prob * magnitude
            x.points = x.points[keep]
        return x


class CutPcAugmentation:
    def __init__(self, prob_cut=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        self._p = prob_cut
        self._scale = scale
        self._ratio = ratio
        self._value = value
        self._inplace = inplace

    def __call__(self, x: KittiFrame, magnitude: float = 1):
        scale = magnitude * self._scale[0], magnitude * self._scale[1]
        random_erasing = RandomErasing(self._p, scale, self._ratio, self._value, self._inplace),
        return random_erasing(x)


class GlobalNoise:
    def __init__(self, prob_aug: float, trans_std: float = 0.2, scale_delta: float = 0.05):
        self._prob_aug = prob_aug
        self._trans_std = trans_std
        self._scale_delta = scale_delta

    def __call__(self, f: KittiFrame):
        noise = np.random.standard_normal((3,)) * self._trans_std
        scale = np.random.uniform(1 - self._scale_delta, 1 + self._scale_delta)

        f.points[:, :3] *= scale
        f.points[:, :3] += noise

        new_labels = []
        for label in f.labels:
            label.location *= scale
            label.dimensions *= scale
            label.location += noise
            new_labels.append(label)

        f.labels = new_labels
        return f


def label_to_array(label: KittiLabel):
    x, y, z = label.location
    l, h, w = label.dimensions
    theta = label.rotation_y
    return [x, y, z, l, h, w, theta]


class BoxNoise:
    def __init__(self,
                 translation_std=None,
                 global_rot_range=None,
                 rot_range=None,
                 num_try=100):
        if translation_std is None:
            translation_std = [0.25, 0.25, 0.25]
        if global_rot_range is None:
            global_rot_range = [0.0, 0.0]
        if rot_range is None:
            rot_range = [-0.15707963267, 0.15707963267]

        self._translation_std = translation_std
        self._global_rot_range = global_rot_range
        self._rot_range = rot_range
        self._num_try = num_try

    def __call__(self, f: KittiFrame):
        gt_bboxes_3d = np.stack([label_to_array(x) for x in f.labels])
        points = f.points

        numpy_box = gt_bboxes_3d.copy()
        swap_indices = [0, 1, 2, 3]
        numpy_points = points.copy()[:, swap_indices]

        noise_per_object_v3_(
            numpy_box,
            numpy_points,
            rotation_perturb=self._rot_range,
            center_noise_std=self._translation_std,
            global_random_rot_range=self._global_rot_range,
            num_try=self._num_try)

        for i, (x, y, z, l, h, w, theta) in enumerate(numpy_box):
            f.labels[i].location = np.array([x, y, z])
            f.labels[i].dimensions = np.array([l, h, w])
            f.labels[i].rotation_y = theta

        f.points = numpy_points.copy()[:, swap_indices]

        return f


@dataclass
class Sample:
    points: np.ndarray
    label: KittiLabel
    camera_label: KittiLabelCamera


class ObjectSample:
    def __init__(self, dataset_root: str, num_sample: int):
        self._sample_path = pathlib.Path(dataset_root).joinpath('samples.pkl')
        # TODO generate samples if not found
        if not self._sample_path.exists():
            raise FileNotFoundError(f'Cannot find samples at {self._sample_path}')
        with open(self._sample_path, 'rb') as f:
            self._samples = pickle.load(f)
        self._num_sample = num_sample

    def __call__(self, f: KittiFrame):
        num_samples = (np.random.randint(0, self._num_sample) + np.random.randint(0,
                                                                                  self._num_sample) + np.random.randint(
            0, self._num_sample)) % self._num_sample
        if num_samples == 0:
            return f

        bbox_avoid_center = np.stack([label_to_array(x) for x in f.labels])
        bbox_avoid = box_np_ops.center_to_corner_box2d(bbox_avoid_center[:, 0:2], bbox_avoid_center[:, 3:5],
                                                       bbox_avoid_center[:, 6])

        samples = []
        for i in range(num_samples):
            s = self._sample_no_collision(bbox_avoid)
            if s is not None:
                samples.append(s)
                b = np.expand_dims(label_to_array(s.label), axis=0)
                b = box_np_ops.center_to_corner_box2d(b[:, 0:2], b[:, 3:5], b[:, 6])
                bbox_avoid = np.concatenate([bbox_avoid, b])

        if len(samples) == 0:
            return f

        sample_labels = [s.label for s in samples]
        sample_camera_labels = [s.camera_label for s in samples]
        sample_bboxes = np.stack([label_to_array(x.label) for x in samples])
        f.labels.extend(sample_labels)
        f.labels_camera.extend(sample_camera_labels)

        # remove points in samples
        masks = box_np_ops.points_in_rbbox(f.points[:, :3], sample_bboxes)
        f.points = f.points[np.logical_not(masks.any(-1))]

        f.points = np.concatenate([f.points, *[s.points for s in samples]], axis=0)

        return f

    def _sample_no_collision(self, bbox_avoid) -> Optional[Sample]:
        num_gt = bbox_avoid.shape[0]
        s = np.random.choice(self._samples, 1)[0]
        box_center = np.expand_dims(np.array(label_to_array(s.label)), axis=0)
        box = box_np_ops.center_to_corner_box2d(box_center[:, 0:2], box_center[:, 3:5], box_center[:, 6])

        total_boxes = np.concatenate([bbox_avoid, box], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_boxes, total_boxes)
        diag = np.arange(total_boxes.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + 1):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
                return None
            else:
                return s
