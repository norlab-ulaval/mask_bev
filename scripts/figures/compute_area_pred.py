import pathlib
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
import yaml
import torch.nn.functional as F

from point_mask.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset, \
    SemanticKittiRawLabel
from point_mask.datasets.semantic_kitti.semantic_kitti_mask_data_module import SemanticKittiMaskDataModule
from point_mask.datasets.semantic_kitti.semantic_kitti_mask_dataset import SemanticKittiMaskDataset
from point_mask.datasets.semantic_kitti.semantic_kitti_rasterizer import SemanticKittiRasterizer
from point_mask.point_mask_module import PointMaskModule

if __name__ == '__main__':
    sample_idx = 0
    pl.seed_everything(123)

    config_path = pathlib.Path('../configs/semantic_kitti/21_point_mask_data_aug_gentle.yml')
    checkpoint = pathlib.Path(
        '/home/william/Datasets/checkpoints/21_point_mask_data_aug_gentle/21_point_mask_data_aug_gentle-epoch=21-val_loss=4.657507.ckpt')

    # Load model
    exp_name = config_path.stem
    checkpoint_folder_path = pathlib.Path('checkpoints').joinpath(exp_name)
    with open(config_path, 'r') as f:
        config: dict = yaml.safe_load(f)
    config['checkpoint'] = checkpoint
    config['shuffle_train'] = False
    config['batch_size'] = 4
    config['num_workers'] = 4

    model = PointMaskModule.from_config(config, exp_name, checkpoint_folder_path)
    datamodule = SemanticKittiMaskDataModule('../data/SemanticKITTI', **config)

    dataloader = datamodule.val_dataloader()
    iterator = iter(dataloader)

    areas = []
    for i in tqdm.tqdm(range(len(dataloader))):
        point_clouds, (labels_gt, masks_gt), metadata = next(iterator)

        # Get mask and encoded point cloud
        point_clouds[0]
        encoded = model.forward_encode(point_clouds)
        features = model.forward_backbone(encoded)
        pred_cls, pred_masks, _ = model.pred_masks(features)
        masks_ = pred_masks[-1]
        masks_ = F.interpolate(masks_, (500, 500))
        masks_ = torch.sigmoid(masks_) > 0.5
        a = masks_.sum(-1).sum(-1).flatten().cpu().numpy().tolist()
        areas.extend(a)


    with open('data/SemanticKITTI/mask_area_pred.pkl', 'wb') as f:
        pickle.dump(areas, f)

