import time
from typing import Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import mask_bev.utils.pipeline as pp
from mask_bev.datasets.apply_transform import ApplyTransform
from mask_bev.datasets.collate_type import CollateType
from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset, \
    SemanticKittiRawLabel
from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_dataset import SemanticKittiMaskDataset
from mask_bev.datasets.semantic_kitti.semantic_kitti_transforms import MaskScanToPointCloud, MaskScanToMask, \
    MaskListCollateHeight, MaskTensorCollate, ShufflePointCloud, MaskToLabelInstanceMasks, FrameMetaData, \
    LabelMaskToMask2FormerLabel, FilterSmallMasks


# TODO Unify with other data modules
class SemanticKittiMaskDataModule(pl.LightningDataModule):
    def __init__(self, root_path: str, batch_size: int, min_num_points, num_queries: int, x_range: (int, int),
                 y_range: (int, int), z_range: (int, int), voxel_size: float, remove_unseen: bool,
                 num_workers: int = 8, pin_memory: bool = True, collate_fn: CollateType = CollateType.ListCollate,
                 shuffle_train: bool = True, dataset_transform: Callable = None, predict_heights: bool = False,
                 head_num_classes: int = 1, min_num_inst_pixels=300, **kwargs):
        """
        Pytorch lightning wrapper around SemanticKittiMaskDataset
        :param root_path: root path of the dataset
        :param batch_size: mini-batch size
        :param min_num_points: minimum number of points to be considered seen
        :param num_workers: number of workers for the data preparation (0 to disable multi-process)
        :param pin_memory: allow the GPU to directly access memory
        :param collate_fn: collate function that collects samples into batchs
        :param dataset_transform: transform to apply to `SemanticKittiMaskDataset`
        :param predict_heights: predict the height of the objects
        :param head_num_classes: number of classes to predict
        :param min_num_inst_pixels: minimum number of pixels to consider an instance
        :param kwargs: additional arguments
        """
        super().__init__()
        self._batch_size = batch_size
        self._min_num_points = min_num_points
        self._num_queries = num_queries
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range
        self._voxel_size = voxel_size
        self._remove_unseen = remove_unseen
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle_train = shuffle_train
        self._dataset_transform = dataset_transform
        self._predict_heights = predict_heights
        self._num_classes = head_num_classes
        self._min_num_inst_pixels = min_num_inst_pixels

        included_labels = [SemanticKittiRawLabel.CAR]

        self._train_seq_dataset = SemanticKittiSequenceDataset(root_path, 'train', included_labels=included_labels)
        self._valid_seq_dataset = SemanticKittiSequenceDataset(root_path, 'valid', included_labels=included_labels)
        self._test_seq_dataset = SemanticKittiSequenceDataset(root_path, 'test', included_labels=included_labels)

        self._train_dataset = SemanticKittiMaskDataset(self._train_seq_dataset, self._x_range, self._y_range,
                                                       self._z_range, self._voxel_size, self._remove_unseen,
                                                       self._min_num_points, transform=self._dataset_transform)
        self._valid_dataset = SemanticKittiMaskDataset(self._valid_seq_dataset, self._x_range, self._y_range,
                                                       self._z_range, self._voxel_size, self._remove_unseen,
                                                       self._min_num_points, transform=self._dataset_transform)
        self._test_dataset = SemanticKittiMaskDataset(self._test_seq_dataset, self._x_range, self._y_range,
                                                      self._z_range, self._voxel_size, self._remove_unseen,
                                                      self._min_num_points, transform=self._dataset_transform)

        transform_to_pair = self._build_transform()
        self._train_dataset = ApplyTransform(self._train_dataset, transform_to_pair)
        self._valid_dataset = ApplyTransform(self._valid_dataset, transform_to_pair)
        self._test_dataset = ApplyTransform(self._test_dataset, MaskScanToPointCloud())
        if collate_fn == CollateType.ListCollate:
            # self._collate_fn = MaskListCollateHeight() if self._predict_heights else MaskListCollate()
            self._collate_fn = MaskListCollateHeight()
        elif collate_fn == CollateType.TensorCollate:
            self._collate_fn = MaskTensorCollate()
        else:
            raise ValueError('Invalid collate_fn')

    @property
    def num_queries(self):
        return self._num_queries

    def _build_transform(self):
        # if self._predict_heights:
        if True:
            return pp.Compose([
                FilterSmallMasks(self._min_num_inst_pixels),
                pp.Tupled(3),
                pp.First(pp.Compose([
                    MaskScanToPointCloud(),
                    ShufflePointCloud(),
                ])),
                pp.Second(pp.Compose([
                    MaskScanToMask(),
                    MaskToLabelInstanceMasks(self._num_queries),
                    LabelMaskToMask2FormerLabel(self._num_classes),
                ])),
                pp.Third(pp.Compose([
                    FrameMetaData(),
                    # FrameRoundedHeight() if self._predict_heights else pp.Identity(),
                ])),
            ])
        else:
            return pp.Compose([
                pp.Tupled(2),
                pp.First(pp.Compose([
                    MaskScanToPointCloud(),
                    ShufflePointCloud(),
                ])),
                pp.Second(pp.Compose([
                    MaskScanToMask(),
                    MaskToLabelInstanceMasks(self._num_queries),
                    LabelMaskToMask2FormerLabel(self._num_classes),
                ])),
            ])

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                          pin_memory=self._pin_memory, shuffle=self._shuffle_train, drop_last=True,
                          collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self._valid_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                          pin_memory=self._pin_memory, shuffle=False, drop_last=True, collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                          pin_memory=self._pin_memory, shuffle=False, drop_last=False, collate_fn=self._collate_fn)


if __name__ == '__main__':
    begin = time.perf_counter()
    x_range, y_range, z_range, voxel_size = (-40, 40), (-40, 40), (-10, 10), 0.16
    batch_size = 8
    datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', batch_size, 1, 40, x_range,
                                             y_range, z_range, voxel_size, True, 0, False,
                                             collate_fn=CollateType.TensorCollate, shuffle_train=False)
    dataloader = datamodule.train_dataloader()

    point_clouds, (labels, masks) = next(iter(dataloader))

    for pc in point_clouds:
        print(len(pc))
    print(time.perf_counter() - begin)
