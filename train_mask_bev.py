import argparse
import os
import subprocess
import sys

import yaml

from mask_bev.augmentations.kitti_mask_augmentations import make_kitti_augmentation_list
from mask_bev.augmentations.semantic_kitti_mask_augmentations import make_semantic_kitti_augmentation_list
from mask_bev.augmentations.waymo_mask_augmentations import make_waymo_augmentation_list
from mask_bev.datasets.kitti.kitti_data_module import KittiDataModule
from mask_bev.datasets.waymo.waymo_data_module import WaymoDataModule
from mask_bev.mask_bev_module import MaskBevModule

os.environ['OMP_NUM_THREADS'] = str(6)

import pathlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import mask_bev.utils.pipeline as pp

from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_data_module import SemanticKittiMaskDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file for current run', required=True)
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    exp_name = config_path.stem
    checkpoint_folder_path = pathlib.Path('checkpoints').joinpath(exp_name)
    if not config_path.exists():
        raise ValueError(f'Could not find config at path {config_path}')
    with open(config_path, 'r') as f:
        config: dict = yaml.safe_load(f)

    model = MaskBevModule.from_config(config, exp_name, checkpoint_folder_path)

    # data
    dataset_name = config.get('dataset', 'semantic-kitti')
    augmentations_list = config.get('augmentations', [])
    if dataset_name == 'semantic-kitti':
        augmentation = pp.Compose(make_semantic_kitti_augmentation_list(augmentations_list))
        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', dataset_transform=augmentation, **config)
    elif dataset_name == 'waymo':
        frame_aug = pp.Compose(make_waymo_augmentation_list(augmentations_list))
        mask_aug = lambda x: x
        datamodule = WaymoDataModule('~/Datasets/Waymo/converted', frame_transform=frame_aug, mask_transform=mask_aug,
                                     **config)
    elif dataset_name == 'kitti':
        frame_aug = pp.Compose(make_kitti_augmentation_list(augmentations_list))
        mask_aug = lambda x: x
        datamodule = KittiDataModule('data/KITTI', frame_transform=frame_aug, mask_transform=mask_aug, **config)
    else:
        raise NotImplementedError(dataset_name)

    # training
    logger = TensorBoardLogger(f'tb_logs/{dataset_name}', name=exp_name)
    limit_train_batches = config.get('limit_train_batches', 1.0)
    limit_val_batches = config.get('limit_val_batches', 1.0)
    log_every_n_steps = config.get('log_every_n_steps', 50)

    check_metric = 'val_loss' if limit_val_batches > 0 else 'train_loss'
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
    available_gpu = list(range(num_gpus))
    print(f'Using GPUs {available_gpu}')
    trainer = pl.Trainer(accelerator='gpu', devices=available_gpu, precision=32, logger=logger,
                         min_epochs=0, max_epochs=1000,
                         log_every_n_steps=log_every_n_steps,
                         limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches,
                         callbacks=[
                             EarlyStopping(check_metric, patience=30, verbose=True),
                             DeviceStatsMonitor(),
                             LearningRateMonitor(),
                             ModelCheckpoint(
                                 monitor=check_metric,
                                 dirpath=str(checkpoint_folder_path),
                                 filename=exp_name + f'-{{epoch:02d}}-{{{check_metric}:.6f}}',
                                 save_top_k=1,
                                 save_last=True,
                                 mode='min',
                             )
                         ])
    # train
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)

    # continue training
    # val_loss_regex = re.compile(r'val_loss=([0-9]+\.?[0-9]*)')
    # checkpoint_path = pathlib.Path('checkpoints')
    # checkpoints = [f for f in checkpoint_path.iterdir() if f.name != 'last.ckpt']
    # best_checkpoint = min(checkpoints, key=lambda x: float(val_loss_regex.search(str(x)).group(1)))
    # print(f'Resuming from {best_checkpoint}')
    # trainer = pl.Trainer(resume_from_checkpoint=best_checkpoint, gpus=[0])
    # trainer.fit(model, datamodule)

    # testing
    # trainer.test(model, datamodule)
    # test_set = TestSetDataLoader('data/test',
    #                              transform=pp.Compose([TextureListToTensorList(), lambda x: torch.cat(x, dim=0)]))
    # test_loader = DataLoader(test_set)
    # trainer.test(model, test_loader)
