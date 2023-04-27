import copy
import numbers
from typing import Union, Tuple, Optional, Dict, List, Callable

import numpy as np
import torch
from cv2 import cv2
from torchvision.transforms import RandomErasing

from mask_bev.augmentations.rand_augment import RandAugment
from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_dataset import SemanticKittiMaskScan


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
        transforms = make_semantic_kitti_augmentation_list(args.get('transforms'))
        return RandAugment(args.get('num_augments'), transforms, args.get('magnitude'))
    elif name == 'cut_pc':
        constructor = CutPcAugmentation
    else:
        raise NotImplementedError(f'{name} is not implemented')
    kwargs = copy.copy(args)
    kwargs.pop('name')
    return constructor(**kwargs)


def make_semantic_kitti_augmentation_list(augmentations: List[Dict]) -> [Callable]:
    return [make_augmentation(aug) for aug in augmentations]


class Flip:
    def __init__(self, prob_flip_x: float = 0.5, prob_flip_y: float = 0.5):
        self._prob_flip_x = prob_flip_x
        self._prob_flip_y = prob_flip_y

    def __call__(self, x: SemanticKittiMaskScan, magnitude: float = 1) -> SemanticKittiMaskScan:
        if np.random.uniform(0, 1) < self._prob_flip_x * magnitude:
            x.scan.point_cloud[:, 0] = -x.scan.point_cloud[:, 0]
            x.mask = x.mask[::-1, :].copy()
        if np.random.uniform(0, 1) < self._prob_flip_y * magnitude:
            x.scan.point_cloud[:, 1] = -x.scan.point_cloud[:, 1]
            x.mask = x.mask[:, ::-1].copy()
        return x


class ShufflePoints:
    def __init__(self, prob_shuffle: float = 0.5):
        self._prob_shuffle = prob_shuffle

    def __call__(self, x: SemanticKittiMaskScan, magnitude: float = 1) -> SemanticKittiMaskScan:
        if np.random.uniform(0, 1) < self._prob_shuffle * magnitude:
            np.random.shuffle(x.scan.point_cloud)
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

    def __call__(self, x: SemanticKittiMaskScan, magnitude: float = 1) -> SemanticKittiMaskScan:
        if np.random.uniform(0, 1) < self._rotate_prob:
            rotation_range = (self._rotation_range[0] * magnitude, self._rotation_range[1] * magnitude)
            theta = np.random.uniform(*rotation_range)

            c = np.cos(np.deg2rad(theta))
            s = np.sin(np.deg2rad(theta))
            R = np.array([[c, -s, 0, 0],
                          [s, c, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

            pc = x.scan.point_cloud
            scene_pc_homo = np.copy(pc)
            scene_pc_homo[:, 3] = 1
            scene_pc_homo = (R @ scene_pc_homo.T).T
            scene_pc_homo /= scene_pc_homo[:, 3].reshape((-1, 1))
            x.scan.point_cloud[:, :3] = scene_pc_homo[:, :3]

            sx, sy = x.mask.shape
            R_2d = cv2.getRotationMatrix2D((sx / 2, sy / 2), theta, 1)
            x.mask = cv2.warpAffine(x.mask, R_2d, x.mask.shape, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        return x


class DecimatePoints:
    def __init__(self, prob_decimate: float, keep_every: int):
        self._prob_decimate = prob_decimate
        self._keep_every = keep_every

    def __call__(self, x: SemanticKittiMaskScan, magnitude: float = 1) -> SemanticKittiMaskScan:
        if np.random.uniform(0, 1) < self._prob_decimate:
            pc = x.scan.point_cloud
            x.scan.point_cloud = pc[torch.randperm(pc.shape[0])][::int(self._keep_every * magnitude)]
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

    def __call__(self, x: SemanticKittiMaskScan, magnitude: float = 1) -> SemanticKittiMaskScan:
        if np.random.uniform(0, 1) < self._prob_jitter:
            noise = np.random.standard_normal(x.scan.point_cloud.shape)
            for dim in range(3):
                noise[:, dim] *= self._jitter_std[dim]
            if self._max_delta is not None:
                for dim in range(3):
                    np.clip(noise[:, dim], -self._max_delta[dim], self._max_delta[dim], noise[:, dim])
            noise[:, 3] *= self._intensity_std
            if self._intensity_max_delta is not None:
                np.clip(noise[:, 3], -self._intensity_max_delta, self._intensity_max_delta, noise[:, 3])

            x.scan.point_cloud += noise * magnitude
            np.clip(x.scan.point_cloud[:, 3], 0, 1, x.scan.point_cloud[:, 3])
        return x


class RandomDropPoints:
    def __init__(self, prob_drop: float, per_point_drop_prob: float):
        self._prob_drop = prob_drop
        self._per_point_drop_prob = per_point_drop_prob

    def __call__(self, x: SemanticKittiMaskScan, magnitude: float = 1) -> SemanticKittiMaskScan:
        if np.random.uniform(0, 1) < self._prob_drop:
            n = x.scan.point_cloud.shape[0]
            keep = np.random.uniform(0, 1, n) >= self._per_point_drop_prob * magnitude
            x.scan.point_cloud = x.scan.point_cloud[keep]
            x.scan.inst_label = x.scan.inst_label[keep]
        return x


class CutPcAugmentation:
    def __init__(self, prob_cut=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        self._p = prob_cut
        self._scale = scale
        self._ratio = ratio
        self._value = value
        self._inplace = inplace

    def __call__(self, x: SemanticKittiMaskScan, magnitude: float = 1):
        scale = magnitude * self._scale[0], magnitude * self._scale[1]
        random_erasing = RandomErasing(self._p, scale, self._ratio, self._value, self._inplace),
        return random_erasing(x)
