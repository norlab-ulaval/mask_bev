from enum import Enum

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, random_split

from mask_bev.datasets.apply_transform import ApplyTransform
from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiRawLabel, SemanticKittiDataset
from mask_bev.datasets.semantic_kitti.semantic_kitti_transforms import ScanToPointCloud, ScanListCollate


class CollateType(Enum):
    TensorCollate = 0
    ListCollate = 1


class SemanticKittiStablePointsDataModule(pl.LightningDataModule):
    def __init__(self, root_path: str, batch_size: int, num_workers: int = 8, pin_memory: bool = True,
                 shuffle: bool = True):
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle = shuffle

        included_labels = [SemanticKittiRawLabel.CAR]

        train_dataset = SemanticKittiDataset(root_path, 'train', included_labels=included_labels)
        valid_dataset = SemanticKittiDataset(root_path, 'valid', included_labels=included_labels)
        test_dataset = SemanticKittiDataset(root_path, 'test', included_labels=included_labels)

        dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
        dataset = ApplyTransform(dataset, ScanToPointCloud())
        self._train_dataset, self._valid_dataset, _ = self.__get_split(dataset)

        self._collate_fn = ScanListCollate()

    @staticmethod
    def __get_split(dataset):
        num_sample = len(dataset)
        test_percent = 0
        train_valid_percent = 1 - test_percent
        train_percent = 0.8
        valid_percent = 1 - train_percent
        return random_split(dataset,
                            [int(np.ceil(num_sample * train_valid_percent * train_percent)),
                             int(np.floor(num_sample * train_valid_percent * valid_percent)),
                             int(np.floor(num_sample * test_percent))])

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                          pin_memory=self._pin_memory, shuffle=self._shuffle, drop_last=True,
                          collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self._valid_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                          pin_memory=self._pin_memory, shuffle=False, drop_last=True, collate_fn=self._collate_fn)
