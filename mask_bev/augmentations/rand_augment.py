import random
from typing import Callable


class RandAugment:
    def __init__(self, num_augments: int, transforms: [Callable], magnitude: float):
        """
        RandAugment from Cubuk et al.
        Magnitude is now a float in [0.5, 1.5]
        :param num_augments: num of augment to apply
        :param transforms: list of transforms to choose from
        :param magnitude: magnitude of each transform
        """
        self._num_augments = num_augments
        self._transforms = transforms
        self._magnitude = magnitude

    def __call__(self, x):
        sampled_transforms = random.choices(self._transforms, k=self._num_augments)
        for trans in sampled_transforms:
            x = trans(x, self._magnitude)
        return x
