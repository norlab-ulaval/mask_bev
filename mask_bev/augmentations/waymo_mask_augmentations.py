import copy
import numbers
from typing import Union, Tuple, Optional, Dict, List, Callable

import numpy as np
import torch
from torch_waymo import SimplifiedFrame
from torch_waymo.protocol.label_proto import Label


def make_augmentation(aug) -> Callable:
    name = aug.get('name')
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
        raise NotImplementedError('rand augment')
    else:
        raise NotImplementedError(f'{name} is not implemented')
    kwargs = copy.copy(aug)
    kwargs.pop('name')
    return constructor(**kwargs)


def make_waymo_augmentation_list(augmentations: List[Dict]) -> [Callable]:
    return [make_augmentation(aug) for aug in augmentations]


class Flip:
    def __init__(self, prob_flip_x: float = 0, prob_flip_y: float = 0.5):
        if prob_flip_x != 0:
            raise ValueError('Cannot flip in x')
        self._prob_flip_x = prob_flip_x
        self._prob_flip_y = prob_flip_y

    def __call__(self, x: SimplifiedFrame) -> SimplifiedFrame:
        if np.random.uniform(0, 1) < self._prob_flip_x:
            ...
        if np.random.uniform(0, 1) < self._prob_flip_y:
            for i in range(len(x.points)):
                x.points[i][:, 1] = -x.points[i][:, 1]
            x.laser_labels = [self._flip_y_label(label) for label in x.laser_labels]
        return x

    def _flip_y_label(self, label):
        args = label.__dict__
        box = label.box
        box.center_y = -box.center_y
        args['box'] = box
        return Label(**args)


class ShufflePoints:
    def __init__(self, prob_shuffle: float = 0.5):
        self._prob_shuffle = prob_shuffle

    def __call__(self, x: SimplifiedFrame) -> SimplifiedFrame:
        if np.random.uniform(0, 1) < self._prob_shuffle:
            for i in range(len(x.points)):
                np.random.shuffle(x.points[i])
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

    def __call__(self, x: SimplifiedFrame) -> SimplifiedFrame:
        if np.random.uniform(0, 1) < self._rotate_prob:
            theta = np.random.uniform(*self._rotation_range)

            c = np.cos(np.deg2rad(theta))
            s = np.sin(np.deg2rad(theta))
            R = np.array([[c, -s, 0, 0],
                          [s, c, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

            for i in range(len(x.points)):
                pc = x.points[i]
                n = pc.shape[0]
                scene_pc_homo = np.zeros((n, 4))
                scene_pc_homo[:, :3] = pc
                scene_pc_homo[:, 3] = 1
                scene_pc_homo = (R @ scene_pc_homo.T).T
                scene_pc_homo /= scene_pc_homo[:, 3].reshape((-1, 1))
                x.points[i][:, :3] = scene_pc_homo[:, :3]

            x.laser_labels = [self._rotate_label(label, theta, R) for label in x.laser_labels]

        return x

    def _rotate_label(self, label, angle, R):
        angle = np.deg2rad(angle)

        box = label.box
        location = np.array([box.center_x, box.center_y, box.center_z, 1])
        location = (R @ location.T).T
        box.center_x, box.center_y, box.center_z, _ = location
        box.heading += angle

        args = label.__dict__
        args['box'] = box
        return Label(**args)


class DecimatePoints:
    def __init__(self, prob_decimate: float, keep_every: int):
        self._prob_decimate = prob_decimate
        self._keep_every = keep_every

    def __call__(self, x: SimplifiedFrame) -> SimplifiedFrame:
        if np.random.uniform(0, 1) < self._prob_decimate:
            for i in range(len(x.points)):
                pc = x.points[i]
                x.points[i] = pc[torch.randperm(pc.shape[0])][::self._keep_every]
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

    def __call__(self, x: SimplifiedFrame) -> SimplifiedFrame:
        if np.random.uniform(0, 1) < self._prob_jitter:
            for i in range(len(x.points)):
                noise = np.random.standard_normal(x.points[i].shape)
                for dim in range(3):
                    noise[:, dim] *= self._jitter_std[dim]
                if self._max_delta is not None:
                    for dim in range(3):
                        np.clip(noise[:, dim], -self._max_delta[dim], self._max_delta[dim], noise[:, dim])
                x.points[i] += noise
        return x


class RandomDropPoints:
    def __init__(self, prob_drop: float, per_point_drop_prob: float):
        self._prob_drop = prob_drop
        self._per_point_drop_prob = per_point_drop_prob

    def __call__(self, x: SimplifiedFrame) -> SimplifiedFrame:
        if np.random.uniform(0, 1) < self._prob_drop:
            for i in range(len(x.points)):
                n = x.points[i].shape[0]
                keep = np.random.uniform(0, 1, n) >= self._per_point_drop_prob
                x.points[i] = x.points[i][keep]
        return x


# TODO make random affine once we have Waymo dataset and a quicker way to generate masks (using bounding boxes)
class RandomAffine:
    def __init__(self):
        ...
