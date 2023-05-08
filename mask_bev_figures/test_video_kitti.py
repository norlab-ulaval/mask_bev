import collections
import functools
import pathlib
import unittest

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
import yaml
from PIL import Image
from matplotlib import pyplot as plt

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiDataset, SemanticKittiSequenceDataset
from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_data_module import SemanticKittiMaskDataModule
from mask_bev.datasets.semantic_kitti.semantic_kitti_rasterizer import SemanticKittiRasterizer
from mask_bev.datasets.semantic_kitti.semantic_kitti_scene import SceneMaker
from mask_bev.mask_bev_module import MaskBevModule
from mask_bev.visualization.point_cloud_viz import show_point_cloud


class TestVideoKITTI(unittest.TestCase):
    def setUp(self):
        self.idx = 2513

    def test_single_scan(self):
        root = 'data/SemanticKITTI'
        train_dataset = SemanticKittiDataset(root, 'valid')
        sample = train_dataset[self.idx]
        scan = sample.point_cloud
        label = sample.sem_label
        color_map = collections.defaultdict(lambda: [0, 0, 0])
        # for i in range(1000):
        #     color_map[i] = [0, 0, 0]
        # color_map[1] = [0, 0, 255]
        color_map = train_dataset.color_map

        scene_maker = SceneMaker(max_points=scan.shape[0])
        scene_maker.add_scan(sample)
        scene = scene_maker.scene
        rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.16, morph_kernel_size=6)
        mask = rasterizer.get_mask_around(sample, scene)

        plt.imshow(mask)
        plt.show()

        path = 'data/output/mask.png'
        mask_img = (mask > 0).astype(np.uint8)
        Image.fromarray(mask_img * 255).save(path)

        show_point_cloud('render', scan, label, color_map, distance=0.25, altitude=np.pi / 4,
                         auto_rotate=True)

    def test_full_map(self):
        root = 'data/SemanticKITTI'
        train_dataset = SemanticKittiSequenceDataset(root, 'valid')

        sequence = train_dataset[0]

        scan_idx = sequence.scan_indices[::10]
        max_points = sum(s.num_points for s in train_dataset.load_scan_indices(scan_idx))
        scene_maker = SceneMaker(max_points=max_points)
        for scan in tqdm.tqdm(train_dataset.load_scan_indices(scan_idx), total=len(scan_idx)):
            scene_maker.add_scan(scan)

        scene = scene_maker.scene
        pc = scene.point_cloud[::5]
        sem = scene.sem_label[::5]

        color_map = collections.defaultdict(lambda: [0, 0, 0])
        # for i in range(1000):
        #     color_map[i] = [0, 0, 0]
        # color_map[1] = [0, 0, 255]
        color_map = train_dataset.dataset.color_map
        sample = train_dataset.dataset[self.idx]
        x, y, z, _ = sample.pose[:, -1]

        scan_positions = sequence.positions()

        rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.16, morph_kernel_size=6)
        mask = rasterizer.get_mask_around(sample, scene)

        plt.imshow(mask)
        plt.show()

        path = './data/output/mask_full.png'
        mask_img = (mask > 0).astype(np.uint8)
        Image.fromarray(mask_img * 255).save(path)

        show_point_cloud('Whole SemanticKittiScene', pc, sem, color_map=color_map, x=z, y=-x, distance=0.25,
                         altitude=np.pi / 4, auto_rotate=True)

    def test_prediction_sequence(self):
        pl.seed_everything(123)

        config_path = pathlib.Path('configs/old_training/semantic_kitti/21_point_mask_data_aug_gentle.yml')
        checkpoint = pathlib.Path(
            '/home/william/Datasets/checkpoints/21_point_mask_data_aug_gentle/21_point_mask_data_aug_gentle-epoch=21-val_loss=4.657507.ckpt')

        # Load model
        exp_name = config_path.stem
        checkpoint_folder_path = pathlib.Path('checkpoints').joinpath(exp_name)
        with open(config_path, 'r') as f:
            config: dict = yaml.safe_load(f)
        config['checkpoint'] = checkpoint
        config['shuffle_train'] = False
        config['batch_size'] = 1
        config['num_workers'] = 0

        model = MaskBevModule.from_config(config, exp_name, checkpoint_folder_path)

        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', **config)
        dataloader = datamodule.val_dataloader()
        iterator = iter(dataloader)

        encoded_path = pathlib.Path('/home/william/Documents/Writing/Videos/enc/')
        pred_path = pathlib.Path('/home/william/Documents/Writing/Videos/pred/')
        raw_mask_path = pathlib.Path('/home/william/Documents/Writing/Videos/raw/')

        gt_path = pathlib.Path('/home/william/Documents/Writing/Videos/gt/')
        gt_path.mkdir(parents=True, exist_ok=True)
        encoded_path.mkdir(parents=True, exist_ok=True)
        pred_path.mkdir(parents=True, exist_ok=True)

        i = -1
        while (data := next(iterator, None)) is not None:
            i += 1
            point_clouds, (labels_gt, masks_gt), _ = data

            # for _ in range(10):
            #     next(iterator, None)

            encoded = model.forward_encode(point_clouds)
            features = model.forward_backbone(encoded)
            pred_cls, pred_masks, _ = model.pred_masks(features)

            # Ground truth
            instances_gt = torch.zeros((500, 500, 1))
            for m_idx, m in enumerate(masks_gt[0]):
                instances_gt[m > 0, :] = 255
            instance_gt_img = instances_gt.cpu().numpy().astype(np.uint8)
            instance_gt_img = cv2.cvtColor(instance_gt_img, cv2.COLOR_GRAY2RGB)
            instance_gt_img = Image.fromarray(instance_gt_img)
            instance_gt_img.save(gt_path.joinpath(f'{i:04}.png'))

            # Encoder
            encoded_img = (model.log_normalize_img(encoded[0]).squeeze().detach().numpy())
            encoded_img *= 255 * 0.4 + 75
            encoded_img = encoded_img.astype(np.uint8)
            encoded_img = cv2.applyColorMap(encoded_img, cv2.COLORMAP_INFERNO)
            encoded_img[np.linalg.norm(encoded_img, axis=2) < 40] = 255
            encoded_img = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB)
            encoded_img = Image.fromarray(np.uint8(encoded_img))
            encoded_img.save(encoded_path.joinpath(f'{i:04}.png'))

            # Masks
            num_detect = 0
            all_masks_sig = []
            print(len(pred_masks[-1][0]))
            for m in range(len(pred_masks[-1][0])):
                try:
                    c = pred_cls[-1][0, m].argmax()
                except:
                    continue
                if c == 1:
                    path = raw_mask_path.joinpath(f'mask_{m:02}')
                    path.mkdir(parents=True, exist_ok=True)

                    log = model.normalize_img(pred_masks[-1][0][m]).detach().cpu().numpy()
                    Image.fromarray(np.uint8(log * 255)).save(path.joinpath(f'raw_{i:04}.png'))

                    mask = model.sigmoid_img(pred_masks[-1][0][m]).unsqueeze(0) > 0.5
                    all_masks_sig.append(mask)
                    num_detect += 1

            if num_detect > 0:
                pred_img = functools.reduce(torch.bitwise_or, all_masks_sig).squeeze().T
            else:
                pred_img = torch.zeros_like(pred_masks[-1][0][0]).squeeze().T

            # Figs
            # fig, axes = plt.subplots(nrows=1, ncols=3)
            # fig.suptitle(f'{i}')
            # axes[0].axis('off')
            # axes[1].axis('off')
            # axes[2].axis('off')
            # axes[0].imshow(encoded_img)
            # axes[1].imshow(instances_gt[:, :])
            # axes[2].imshow(pred_img.T > 0, cmap='gray')
            # fig.tight_layout()
            # fig.show()

            pred_img = Image.fromarray(np.uint8(pred_img.detach().cpu().numpy().T * 255))
            pred_img.save(pred_path.joinpath(f'{i:04}.png'))

    def test_kitti_seq(self):
        pl.seed_everything(123)

        config_path = pathlib.Path('configs/old_training/semantic_kitti/21_point_mask_data_aug_gentle.yml')

        with open(config_path, 'r') as f:
            config: dict = yaml.safe_load(f)
        config['shuffle_train'] = False
        config['batch_size'] = 1
        config['num_workers'] = 0

        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', **config)
        dataloader = datamodule.val_dataloader()

        gt_path = pathlib.Path(f'data/output/gt/')
        gt_path.mkdir(parents=True, exist_ok=True)

        def every_nth(iterator, n):
            for i, item in enumerate(iterator):
                if i % n == 0:
                    yield item

        for scan_num, (point_clouds, (labels_gt, masks_gt), _) in tqdm.tqdm(enumerate(every_nth(dataloader, 10))):
            # Ground truth
            instances_gt = torch.zeros((500, 500, 1))
            for m_idx, m in enumerate(masks_gt[0]):
                instances_gt[m > 0, :] = 255
            instance_gt_img = instances_gt.cpu().numpy().astype(np.uint8)
            instance_gt_img = cv2.cvtColor(instance_gt_img, cv2.COLOR_GRAY2RGB)
            instance_gt_img = Image.fromarray(instance_gt_img)
            instance_gt_img.save(gt_path.joinpath(f'{scan_num:04}.png'))
