import functools
import pathlib
import tracemalloc
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from PIL import Image
from matplotlib import colors

from mask_bev.datasets.kitti.kitti_data_module import KittiDataModule
from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_data_module import SemanticKittiMaskDataModule
from mask_bev.mask_bev_module import MaskBevModule

tracemalloc.start()


# TODO clean up
class TestFigures(unittest.TestCase):
    def test_semantic_kitti_2(self):
        plt.rcParams['figure.dpi'] = 450
        sample_idx = 0
        pl.seed_everything(123)

        config_path = pathlib.Path('configs/semantic_kitti/21_point_mask_data_aug_gentle.yml')
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
        gen_figs = True

        if gen_figs:
            dataloader = datamodule.val_dataloader()
            iterator = iter(dataloader)
            for j in range(20):
                # some good
                # for i in range(23):
                # some bad
                # for i in range(18):
                # for i in range(17):
                # for i in range(15):
                # for i in range(13):
                # for i in range(50):
                # for _ in range(4):
                point_clouds, (labels_gt, masks_gt), _ = next(iterator)
                pathlib.Path(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}').mkdir(
                    exist_ok=True)
                instances_gt = torch.zeros((500, 500, 3))
                for i, m in enumerate(masks_gt[0]):
                    r = np.random.uniform(0, 1)
                    g = np.random.uniform(0, 1)
                    b = np.random.uniform(0, 1)
                    instances_gt[m > 0, :] = 1

                # Get mask and encoded point cloud
                point_clouds[0]
                encoded = model.forward_encode(point_clouds)
                features = model.forward_backbone(encoded)
                pred_cls, pred_masks, _ = model.pred_masks(features)

                # Prepare images
                encoded_img = (model.log_normalize_img(encoded[0]).squeeze().detach().numpy())

                encoded_img *= 255 * 0.4 + 75
                encoded_img = encoded_img.astype(np.uint8)
                img = cv2.applyColorMap(encoded_img, cv2.COLORMAP_INFERNO)
                img[np.linalg.norm(img, axis=2) < 40] = 255
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                Image.fromarray(np.uint8(img)).save(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}/enc_2.png')
                plt.imshow(img)
                plt.show()

                gt_img = Image.fromarray(np.uint8(instances_gt[:, :] * 255))
                w, h = gt_img.size
                # gt_img = gt_img.resize((w * 2, h * 2)).crop((w / 2, h / 2, 3 * w / 2, 3 * h / 2))
                gt_img.save(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}/gt_2.png')

                all_masks_sig = []
                num_detect = 0
                mask_num = 0
                for i in range(len(pred_masks[-1][0])):
                    c = pred_cls[-1][0][i].argmax()
                    if c == 1:
                        log = model.normalize_img(pred_masks[-1][0][i]).detach().cpu().numpy()
                        # plt.imshow(log, cmap='gray')
                        # plt.show()

                        Image.fromarray(np.uint8(log * 255)).save(
                            f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}/{mask_num}.png')
                        mask_num += 1

                        mask = model.sigmoid_img(pred_masks[-1][0][i]).unsqueeze(0) > 0.5
                        all_masks_sig.append(mask)
                        num_detect += 1
                print(num_detect)
                pred = functools.reduce(torch.bitwise_or, all_masks_sig).squeeze().T

                # plt.imshow(pred)
                # plt.show()

                # Flip masks
                # pred = torch.flip(pred, dims=(0,))
                # instances_gt = torch.flip(instances_gt, dims=(0,))

                # Plot
                fig, axes = plt.subplots(nrows=1, ncols=3)

                # axes[0].set_xlabel('y')
                # axes[0].set_ylabel('x')
                fig.suptitle(f'{j}')
                axes[0].axis('off')
                axes[1].axis('off')
                axes[2].axis('off')
                axes[0].imshow(encoded_img)
                axes[1].imshow(instances_gt[:, :])
                axes[2].imshow(pred.T > 0, cmap='gray')

                pred_img = Image.fromarray(np.uint8(pred.detach().cpu().numpy().T * 255))
                w, h = pred_img.size
                # pred_img = pred_img.resize((2 * w, 2 * h))
                pred_img.save(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}/pred_2.png')

                fig.tight_layout()
                # fig.savefig('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/fig-1.jpg',
                #             bbox_inches='tight')
                # for i in range(5):
                #     plt.imshow(masks_gt[i])
                #     plt.show()
                fig.show()
        trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=32)
        # train
        # trainer.validate(model, datamodule)

        # box_labels = np.stack(
        #     [[*b.location,
        #       *b.dimensions,
        #       b.rotation_y]
        #      for b
        #      in labels])
        #
        # show_point_cloud('render', point_clouds[0][::5], box_labels)

    def test_kitti_seq(self):
        plt.rcParams['figure.dpi'] = 450
        sample_idx = 0
        pl.seed_everything(45)

        config_path = pathlib.Path('configs/kitti/02_kitti_point_mask_lower_lr_no_abs.yml')
        checkpoint = pathlib.Path(
            '/home/william/Datasets/checkpoints/72_kitti_point_mask_lower_lr_no_abs/72_kitti_point_mask_lower_lr_no_abs-epoch=63-val_loss=4.438325.ckpt')

        # Load model
        exp_name = config_path.stem
        checkpoint_folder_path = pathlib.Path('checkpoints').joinpath(exp_name)
        with open(config_path, 'r') as f:
            config: dict = yaml.safe_load(f)
        config['checkpoint'] = checkpoint
        config['shuffle_train'] = True
        config['batch_size'] = 1
        config['num_workers'] = 0

        model = PointMaskModule.from_config(config, exp_name, checkpoint_folder_path)

        datamodule = KittiDataModule('data/KITTI', **config)
        gen_figs = True

        if gen_figs:
            dataloader = datamodule.val_dataloader()
            iterator = iter(dataloader)
            for j in range(25):
                # for i in range(20):
                point_clouds, (labels_gt, masks_gt), metadata = next(iterator)

                instances_gt = torch.zeros((500, 500, 3))
                for i, m in enumerate(masks_gt[0]):
                    r = np.random.uniform(0, 1)
                    g = np.random.uniform(0, 1)
                    b = np.random.uniform(0, 1)
                    instances_gt[m > 0, :] = 1

                # Get mask and encoded point cloud
                point_clouds[0]
                encoded = model.forward_encode(point_clouds)
                features = model.forward_backbone(encoded)
                pred_cls, pred_masks, _ = model.pred_masks(features)

                # Prepare images
                encoded_img = (model.log_normalize_img(encoded[0]).squeeze().detach().numpy())

                encoded_img *= 255 * 0.4 + 75
                encoded_img = encoded_img.astype(np.uint8)
                img = cv2.applyColorMap(encoded_img, cv2.COLORMAP_INFERNO)
                img[np.linalg.norm(img, axis=2) < 40] = 255
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                Image.fromarray(np.uint8(img)).save(
                    '/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/kitti_enc.png')
                # plt.imshow(img)
                # plt.show()

                gt_img = Image.fromarray(np.uint8(instances_gt[:, :] * 255))
                w, h = gt_img.size
                # gt_img = gt_img.resize((w * 2, h * 2)).crop((w / 2, h / 2, 3 * w / 2, 3 * h / 2))
                gt_img.save(
                    '/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/kitti_gt.png')

                all_masks_sig = []
                num_detect = 0
                mask_num = 0
                for i in range(len(pred_masks[-1][0])):
                    c = pred_cls[-1][0][i].argmax()
                    if c == 1:
                        log = model.normalize_img(pred_masks[-1][0][i]).detach().cpu().numpy()
                        # plt.imshow(log, cmap='gray')
                        # plt.show()

                        Image.fromarray(np.uint8(log * 255)).save(
                            f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/kitti_mask_{mask_num}.png')
                        mask_num += 1

                        mask = model.sigmoid_img(pred_masks[-1][0][i]).unsqueeze(0) > 0.5
                        all_masks_sig.append(mask)
                        num_detect += 1
                print(num_detect)
                if num_detect < 1:
                    continue
                pred = functools.reduce(torch.bitwise_or, all_masks_sig).squeeze().T

                # plt.imshow(pred)
                # plt.show()

                # Flip masks
                # pred = torch.flip(pred, dims=(0,))
                # instances_gt = torch.flip(instances_gt, dims=(0,))

                # Plot
                fig, axes = plt.subplots(nrows=1, ncols=3)

                # axes[0].set_xlabel('y')
                # axes[0].set_ylabel('x')
                axes[0].axis('off')
                axes[1].axis('off')
                axes[2].axis('off')
                axes[0].imshow(encoded_img)
                axes[1].imshow(instances_gt[:, :])
                axes[2].imshow(pred.T > 0, cmap='gray')
                fig.suptitle(f'{j}')

                pred_img = Image.fromarray(np.uint8(pred.detach().cpu().numpy().T * 255))
                w, h = pred_img.size
                # pred_img = pred_img.resize((2 * w, 2 * h))
                pred_img.save(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/kitti_pred.png')

                fig.tight_layout()
                # fig.savefig('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/fig-1.jpg',
                #             bbox_inches='tight')
                # for i in range(5):
                #     plt.imshow(masks_gt[i])
                #     plt.show()
                fig.show()
                if j == 22:
                    break

        trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=32)
        # train
        # trainer.validate(model, datamodule)
        # show_point_cloud('render', point_clouds[0][::5])#, label[::5], color_map, distance=0.4, azimuth=-np.pi / 2 + np.deg2rad(3))

    def test_kitti(self):
        plt.rcParams['figure.dpi'] = 450
        sample_idx = 0
        pl.seed_everything(45)

        config_path = pathlib.Path('configs/training/kitti/02_kitti_point_mask_lower_lr_no_abs.yml')
        checkpoint = pathlib.Path(
            '/home/william/Datasets/checkpoints/72_kitti_point_mask_lower_lr_no_abs/72_kitti_point_mask_lower_lr_no_abs-epoch=63-val_loss=4.438325.ckpt')

        # Load model
        exp_name = config_path.stem
        checkpoint_folder_path = pathlib.Path('checkpoints').joinpath(exp_name)
        with open(config_path, 'r') as f:
            config: dict = yaml.safe_load(f)
        config['checkpoint'] = checkpoint
        config['shuffle_train'] = True
        config['batch_size'] = 1
        config['num_workers'] = 0

        model = MaskBevModule.from_config(config, checkpoint_folder_path)

        datamodule = KittiDataModule('data/KITTI', **config)
        gen_figs = True

        if gen_figs:
            dataloader = datamodule.val_dataloader()
            iterator = iter(dataloader)
            for j in range(25):
                # for i in range(20):
                point_clouds, (labels_gt, masks_gt), metadata = next(iterator)

                instances_gt = torch.zeros((500, 500, 3))
                for i, m in enumerate(masks_gt[0]):
                    r = np.random.uniform(0, 1)
                    g = np.random.uniform(0, 1)
                    b = np.random.uniform(0, 1)
                    instances_gt[m > 0, :] = 1

                # Get mask and encoded point cloud
                point_clouds[0]
                encoded = model.forward_encode(point_clouds)
                features = model.forward_backbone(encoded)
                pred_cls, pred_masks, _ = model.pred_masks(features)

                # Prepare images
                encoded_img = (model.log_normalize_img(encoded[0]).squeeze().detach().numpy())

                encoded_img *= 255 * 0.4 + 75
                encoded_img = encoded_img.astype(np.uint8)
                img = cv2.applyColorMap(encoded_img, cv2.COLORMAP_INFERNO)
                img[np.linalg.norm(img, axis=2) < 40] = 255
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                Image.fromarray(np.uint8(img)).save(
                    '/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/kitti_enc.png')
                # plt.imshow(img)
                # plt.show()

                gt_img = Image.fromarray(np.uint8(instances_gt[:, :] * 255))
                w, h = gt_img.size
                # gt_img = gt_img.resize((w * 2, h * 2)).crop((w / 2, h / 2, 3 * w / 2, 3 * h / 2))
                gt_img.save(
                    '/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/kitti_gt.png')

                all_masks_sig = []
                num_detect = 0
                mask_num = 0
                for i in range(len(pred_masks[-1][0])):
                    c = pred_cls[-1][0][i].argmax()
                    if c == 1:
                        log = model.normalize_img(pred_masks[-1][0][i]).detach().cpu().numpy()
                        # plt.imshow(log, cmap='gray')
                        # plt.show()

                        Image.fromarray(np.uint8(log * 255)).save(
                            f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/kitti_mask_{mask_num}.png')
                        mask_num += 1

                        mask = model.sigmoid_img(pred_masks[-1][0][i]).unsqueeze(0) > 0.5
                        all_masks_sig.append(mask)
                        num_detect += 1
                print(num_detect)
                if num_detect < 5:
                    continue
                pred = functools.reduce(torch.bitwise_or, all_masks_sig).squeeze().T

                # plt.imshow(pred)
                # plt.show()

                # Flip masks
                # pred = torch.flip(pred, dims=(0,))
                # instances_gt = torch.flip(instances_gt, dims=(0,))

                # Plot
                fig, axes = plt.subplots(nrows=1, ncols=3)

                # axes[0].set_xlabel('y')
                # axes[0].set_ylabel('x')
                axes[0].axis('off')
                axes[1].axis('off')
                axes[2].axis('off')
                axes[0].imshow(encoded_img)
                axes[1].imshow(instances_gt[:, :])
                axes[2].imshow(pred.T > 0, cmap='gray')
                fig.suptitle(f'{j}')

                pred_img = Image.fromarray(np.uint8(pred.detach().cpu().numpy().T * 255))
                w, h = pred_img.size
                # pred_img = pred_img.resize((2 * w, 2 * h))
                pred_img.save(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/kitti_pred.png')

                fig.tight_layout()
                # fig.savefig('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/fig-1.jpg',
                #             bbox_inches='tight')
                # for i in range(5):
                #     plt.imshow(masks_gt[i])
                #     plt.show()
                fig.show()
                if j == 22:
                    break

        trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=32)
        # train
        # trainer.validate(model, datamodule)
        # show_point_cloud('render', point_clouds[0][::5])#, label[::5], color_map, distance=0.4, azimuth=-np.pi / 2 + np.deg2rad(3))

    def test_semantic_kitti(self):
        plt.rcParams['figure.dpi'] = 450
        sample_idx = 0
        pl.seed_everything(123)

        config_path = pathlib.Path('configs/semantic_kitti/21_point_mask_data_aug_gentle.yml')
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

        model = PointMaskModule.from_config(config, exp_name, checkpoint_folder_path)

        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', **config)
        gen_figs = False

        if gen_figs:
            dataloader = datamodule.val_dataloader()
            iterator = iter(dataloader)
            for j in range(20):
                # some good
                # for i in range(23):
                # some bad
                # for i in range(18):
                # for i in range(17):
                # for i in range(15):
                # for i in range(13):
                # for i in range(50):
                for _ in range(4):
                    point_clouds, (labels_gt, masks_gt), _ = next(iterator)
                pathlib.Path(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}').mkdir(
                    exist_ok=True)
                instances_gt = torch.zeros((500, 500, 3))
                for i, m in enumerate(masks_gt[0]):
                    r = np.random.uniform(0, 1)
                    g = np.random.uniform(0, 1)
                    b = np.random.uniform(0, 1)
                    instances_gt[m > 0, :] = 1

                # Get mask and encoded point cloud
                point_clouds[0]
                encoded = model.forward_encode(point_clouds)
                features = model.forward_backbone(encoded)
                pred_cls, pred_masks, _ = model.pred_masks(features)

                # Prepare images
                encoded_img = (model.log_normalize_img(encoded[0]).squeeze().detach().numpy())

                encoded_img *= 255 * 0.4 + 75
                encoded_img = encoded_img.astype(np.uint8)
                img = cv2.applyColorMap(encoded_img, cv2.COLORMAP_INFERNO)
                img[np.linalg.norm(img, axis=2) < 40] = 255
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                Image.fromarray(np.uint8(img)).save(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}/enc_2.png')
                # plt.imshow(img)
                # plt.show()

                gt_img = Image.fromarray(np.uint8(instances_gt[:, :] * 255))
                w, h = gt_img.size
                # gt_img = gt_img.resize((w * 2, h * 2)).crop((w / 2, h / 2, 3 * w / 2, 3 * h / 2))
                gt_img.save(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}/gt_2.png')

                all_masks_sig = []
                num_detect = 0
                mask_num = 0
                for i in range(len(pred_masks[-1][0])):
                    c = pred_cls[-1][0][i].argmax()
                    if c == 1:
                        log = model.normalize_img(pred_masks[-1][0][i]).detach().cpu().numpy()
                        # plt.imshow(log, cmap='gray')
                        # plt.show()

                        Image.fromarray(np.uint8(log * 255)).save(
                            f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}/{mask_num}.png')
                        mask_num += 1

                        mask = model.sigmoid_img(pred_masks[-1][0][i]).unsqueeze(0) > 0.5
                        all_masks_sig.append(mask)
                        num_detect += 1
                print(num_detect)
                pred = functools.reduce(torch.bitwise_or, all_masks_sig).squeeze().T

                # plt.imshow(pred)
                # plt.show()

                # Flip masks
                # pred = torch.flip(pred, dims=(0,))
                # instances_gt = torch.flip(instances_gt, dims=(0,))

                # Plot
                fig, axes = plt.subplots(nrows=1, ncols=3)

                # axes[0].set_xlabel('y')
                # axes[0].set_ylabel('x')
                fig.suptitle(f'{j}')
                axes[0].axis('off')
                axes[1].axis('off')
                axes[2].axis('off')
                axes[0].imshow(encoded_img)
                axes[1].imshow(instances_gt[:, :])
                axes[2].imshow(pred.T > 0, cmap='gray')

                pred_img = Image.fromarray(np.uint8(pred.detach().cpu().numpy().T * 255))
                w, h = pred_img.size
                # pred_img = pred_img.resize((2 * w, 2 * h))
                pred_img.save(
                    f'/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/mask_{j}/pred_2.png')

                fig.tight_layout()
                # fig.savefig('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/fig-1.jpg',
                #             bbox_inches='tight')
                # for i in range(5):
                #     plt.imshow(masks_gt[i])
                #     plt.show()
                fig.show()
        trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=32)
        # train
        trainer.validate(model, datamodule)

        # box_labels = np.stack(
        #     [[*b.location,
        #       *b.dimensions,
        #       b.rotation_y]
        #      for b
        #      in labels])
        #
        # show_point_cloud('render', point_clouds[0][::5], box_labels)

    def test_fig_one(self):
        plt.rcParams['figure.dpi'] = 450
        sample_idx = 0
        pl.seed_everything(123)

        config_path = pathlib.Path('configs/waymo/05_waymo_point_mask_data_aug_gentle_lower_lr.yml')
        checkpoint = pathlib.Path(
            '/home/william/Datasets/checkpoints/05_waymo_point_mask_data_aug_gentle_lower_lr/last.ckpt')

        gen_figs = True

        # Load model
        exp_name = config_path.stem
        checkpoint_folder_path = pathlib.Path('checkpoints').joinpath(exp_name)
        with open(config_path, 'r') as f:
            config: dict = yaml.safe_load(f)
        config['checkpoint'] = checkpoint
        config['shuffle_train'] = False
        config['batch_size'] = 1
        config['num_workers'] = 0

        model = PointMaskModule.from_config(config, exp_name, checkpoint_folder_path)

        # Load data
        datamodule = WaymoDataModule('http://127.0.0.1:5000', **config)
        dataloader = datamodule.train_dataloader()
        iterator = iter(dataloader)

        for i in range(200):
            point_clouds, (labels_gt, masks_gt), metadata = next(iterator)
        # instances_gt = model.masks_to_instance_map(masks_gt[0]).squeeze().T

        if gen_figs:
            instances_gt = torch.zeros((500, 500, 3))
            for i, m in enumerate(masks_gt[0]):
                r = np.random.uniform(0, 1)
                g = np.random.uniform(0, 1)
                b = np.random.uniform(0, 1)
                instances_gt[m > 0, :] = torch.tensor([r, g, b])

            # Get mask and encoded point cloud
            point_clouds[0] *= 2
            encoded = model.forward_encode(point_clouds)
            features = model.forward_backbone(encoded)
            pred_cls, pred_masks, _ = model.pred_masks(features)

            # Prepare images
            encoded_img = model.log_normalize_img(encoded[0]).squeeze().detach().numpy()

            encoded_img *= 255 * 0.4 + 75
            # encoded_img[encoded_img > 125] = 125
            encoded_img = encoded_img.astype(np.uint8)
            img = cv2.applyColorMap(encoded_img, cv2.COLORMAP_INFERNO)
            img[(img == img[0, 0])] = 255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(np.uint8(img)).save(
                '/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/pc_close.png')
            plt.imshow(img)
            plt.show()

            gt_img = Image.fromarray(np.uint8(instances_gt[:, :] * 255))
            w, h = gt_img.size
            gt_img = gt_img.resize((w * 2, h * 2)).crop((w / 2, h / 2, 3 * w / 2, 3 * h / 2))
            gt_img.save(
                '/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/pc_close_masks.png')

            all_masks_sig = []
            for i in range(len(pred_masks[-1][0])):
                c = pred_cls[-1][0][i].argmax()
                if c == 1:
                    mask = model.sigmoid_img(pred_masks[-1][0][i]).unsqueeze(0) > 0.5
                    all_masks_sig.append(mask)
            pred = functools.reduce(torch.bitwise_or, all_masks_sig).squeeze().T

            plt.imshow(pred)
            plt.show()

            # Flip masks
            # pred = torch.flip(pred, dims=(0,))
            # instances_gt = torch.flip(instances_gt, dims=(0,))

            # Plot
            fig, axes = plt.subplots(nrows=1, ncols=2)

            # axes[0].set_xlabel('y')
            # axes[0].set_ylabel('x')
            axes[0].axis('off')
            axes[1].axis('off')
            # axes[2].axis('off')
            axes[0].imshow(encoded_img.T)
            axes[1].imshow(instances_gt[:, :])
            # axes[2].imshow(pred.T > 0, cmap='gray')

            fig.tight_layout()
            fig.savefig('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/fig-1.jpg',
                        bbox_inches='tight')
            # for i in range(5):
            #     plt.imshow(masks_gt[i])
            #     plt.show()
            fig.show()

        labels: [Label] = metadata[0]['laser_labels']
        labels = list(filter(lambda label: label.type == Type.TYPE_VEHICLE, labels))
        boxes: [Box] = [label.box for label in labels]
        box_labels = np.stack(
            [[b.center_x, b.center_y, b.center_z, b.length, b.width, b.height, b.heading] for b in boxes])
        show_point_cloud('render', point_clouds[0][::5], box_labels=box_labels)
        # , label[::5], color_map, distance=0.4, azimuth=-np.pi / 2 + np.deg2rad(3))

    def test_fig_pc(self):
        plt.rcParams['figure.dpi'] = 300
        pl.seed_everything(123)

        config_path = pathlib.Path('configs/waymo/05_waymo_point_mask_data_aug_gentle_lower_lr.yml')
        # checkpoint = pathlib.Path(
        #     '/home/william/Datasets/checkpoints/05_waymo_point_mask_data_aug_gentle_lower_lr/last.ckpt')

        # Load model
        exp_name = config_path.stem
        checkpoint_folder_path = pathlib.Path('checkpoints').joinpath(exp_name)
        with open(config_path, 'r') as f:
            config: dict = yaml.safe_load(f)
        # config['checkpoint'] = checkpoint
        config['shuffle_train'] = True
        config['batch_size'] = 1
        # config['voxel_size'] = 0.1

        model = PointMaskModule.from_config(config, exp_name, checkpoint_folder_path)

        # Load data
        datamodule = WaymoDataModule('http://127.0.0.1:5000', **config)
        dataloader = datamodule.train_dataloader()
        iterator = iter(dataloader)

        for sample_idx in range(6, 11):
            point_clouds, (labels_gt, masks_gt) = next(iterator)
            instances_gt = model.masks_to_instance_map(masks_gt[0]).squeeze().T

            # Get mask and encoded point cloud
            encoded = model.forward_encode(point_clouds)
            features = model.forward_backbone(encoded)
            pred_cls, pred_masks = model.pred_masks(features)

            # Prepare images
            encoded_img = model.log_normalize_img(encoded[0]).squeeze().detach().numpy().T
            all_masks_sig = []
            for i in range(len(pred_masks[-1][0])):
                c = pred_cls[-1][0][i].argmax()
                if c == 1:
                    mask = model.sigmoid_img(pred_masks[-1][0][i]).unsqueeze(0) > 0.5
                    all_masks_sig.append(mask)
            # pred = functools.reduce(torch.bitwise_or, all_masks_sig).squeeze().T

            # Flip masks
            # pred = torch.flip(pred, dims=(0,))
            # instances_gt = torch.flip(instances_gt, dims=(0,))

            # Plot
            fig, axes = plt.subplots(nrows=1, ncols=1)

            axes.axis('off')

            def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
                new_cmap = colors.LinearSegmentedColormap.from_list(
                    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                    cmap(np.linspace(minval, maxval, n)))
                return new_cmap

            cmap = truncate_colormap(plt.get_cmap('inferno'), 0.2, 0.6)
            # axes.imshow(encoded_img.T - encoded_img[0, 0], cmap=cmap)
            encoded_img *= 255 * 0.4 + 75
            # encoded_img[encoded_img > 125] = 125
            encoded_img = encoded_img.T.astype(np.uint8)
            img = cv2.applyColorMap(encoded_img, cv2.COLORMAP_INFERNO)
            img[(img == img[0, 0])] = 255
            axes.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            fig.tight_layout()
            if sample_idx == 7:
                # fig.savefig('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/pc.jpg',
                #             bbox_inches='tight')
                break
            fig.show()

    def test_3d(self):
        dataset = WaymoDataset('http://127.0.0.1:5000', 'train', True, 'data/Waymo/cache')

        frame = dataset[150]
        # frame = dataset[1800]
        # exit(0)

        laser_idx = 0
        pc = frame.points[laser_idx]
        # laser_name = LaserName.reverse_dict()[frame.points[laser_idx].name]
        labels: [Label] = frame.laser_labels
        labels = list(filter(lambda label: label.type == Type.TYPE_VEHICLE, labels))
        boxes: [Box] = [label.box for label in labels]
        box_labels = np.stack(
            [[b.center_x, b.center_y, b.center_z, b.length, b.width, b.height, b.heading] for b in boxes])

        def _make_box_vertices_and_indices(boxes: np.ndarray) -> (np.ndarray, np.ndarray):
            num_boxes = boxes.shape[0]
            points_per_box = 8
            triangles_per_box = 12
            points = np.zeros((points_per_box * num_boxes, 3), dtype=np.float32)
            indices = np.zeros((triangles_per_box * num_boxes, 3), dtype=np.uintc)
            for i, [cx, cy, cz, l, w, h, theta] in enumerate(boxes):
                base_point_index = i * points_per_box
                base_indices_index = i * triangles_per_box
                dl, dw = l / 2, w / 2

                center = [cx, cy, cz]
                d = np.array([np.cos(theta), np.sin(theta), 0])
                phi = theta + np.pi / 2
                d_bar = np.array([np.cos(phi), np.sin(phi), 0])
                z = np.array([0, 0, 1])

                points[base_point_index + 0, :] = d * dl + d_bar * dw
                points[base_point_index + 1, :] = - d * dl + d_bar * dw
                points[base_point_index + 2, :] = - d * dl - d_bar * dw
                points[base_point_index + 3, :] = d * dl - d_bar * dw

                for j in range(4):
                    points[base_point_index + 4 + j] = points[base_point_index + j] + h * z

                # center points
                points[base_point_index:base_point_index + 8] += center

                num_faces = 0
                # side faces
                for j in range(4):
                    indices[base_indices_index + num_faces] = [base_point_index + j, (j + 1) % 4 + base_point_index,
                                                               base_point_index + j + 4]
                    num_faces += 1
                    indices[base_indices_index + num_faces] = [(j + 1) % 4 + base_point_index,
                                                               base_point_index + (j + 1) % 4 + 4,
                                                               j + 4 + base_point_index]
                    num_faces += 1

                # bottom face
                indices[base_indices_index + num_faces] = [base_point_index, base_point_index + 3, base_point_index + 1]
                num_faces += 1
                indices[base_indices_index + num_faces] = [base_point_index + 1, base_point_index + 3,
                                                           base_point_index + 2]
                num_faces += 1

                # top face
                indices[base_indices_index + num_faces] = [base_point_index, base_point_index + 3, base_point_index + 1]
                indices[base_indices_index + num_faces] += 4
                num_faces += 1
                indices[base_indices_index + num_faces] = [base_point_index + 1, base_point_index + 3,
                                                           base_point_index + 2]
                indices[base_indices_index + num_faces] += 4
                num_faces += 1

            return points, indices

        def divide_chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        def get_area(a, b, c):
            ax, ay = a
            bx, by = b
            cx, cy = c
            return abs((bx * ay - ax * by) + (cx * by - bx * cy) + (ax * cy - cx * ay)) / 2

        def is_in_rect(p, a, b, c, d):
            area_rect = get_area(a, b, c) + get_area(a, c, d)
            point_area = get_area(a, p, d) + get_area(d, p, c) + get_area(c, p, b) + get_area(p, b, a)
            return point_area > area_rect

        boxes_points, boxes_indices = _make_box_vertices_and_indices(box_labels)

        label = np.zeros((pc.shape[0]), dtype=int)
        scale = 1000
        # for i in range(pc.shape[0]):
        #     px, py, pz = pc[i][:3]
        #     if pz < -0.25:
        #         continue
        #     # label[i] = 1
        #     for b in divide_chunks(boxes_points, 8):
        #         b = np.intp(b[:4, :2] * scale)
        #         if cv2.pointPolygonTest(b, (px * scale, py * scale), False) > 0:
        #             label[i] = 1

        show_point_cloud(f'Waymo sample - Laser', pc, color_map={0: [0, 0, 0], 1: [0, 0, 255]}, label=label,
                         box_labels=box_labels, azimuth=np.pi, altitude=0.01,
                         x=0.14477456146872966, y=-0.1315676315593811, z=0.0, distance=0.11999999999999968)

    def test_crop_pc(self):
        pc_path = '/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/pc.jpg'
        img = Image.open(pc_path)
        w, h = img.size
        offset = 125
        img = img.crop((offset, offset, w - offset, h - offset))
        # img.save('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/pc_cropped.jpg')
        img = img.resize((1000, 1000), resample=Image.BOX)
        # img.save('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/pc_cropped.jpg')

    def test_pc_3d(self):
        root = 'data/SemanticKITTI'
        train_dataset = SemanticKittiDataset(root, 'train')
        scan = train_dataset[0].point_cloud
        label = train_dataset[0].sem_label
        color_map = train_dataset.color_map

        show_point_cloud('render', scan[::5], label[::5], color_map, distance=0.4, azimuth=-np.pi / 2 + np.deg2rad(3))

    def test_mask_scan_fig(self):
        sample_idx = 0
        config_path = pathlib.Path('configs/semantic_kitti/21_point_mask_data_aug_gentle.yml')
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

        model = PointMaskModule.from_config(config, exp_name, checkpoint_folder_path)

        # Load data
        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', **config)
        dataloader = datamodule.train_dataloader()
        for _ in range(sample_idx + 1):
            point_clouds, (labels_gt, masks_gt), metadata = next(iter(dataloader))
        instances_gt = model.masks_to_instance_map(masks_gt[0]).squeeze().T

        # Get mask and encoded point cloud
        encoded = model.forward_encode(point_clouds)
        features = model.forward_backbone(encoded)
        pred_cls, pred_masks, _ = model.pred_masks(features)

        # Prepare images
        encoded_img = model.log_normalize_img(encoded[0]).squeeze().detach().numpy().T
        all_masks_sig = []
        for i in range(len(pred_masks[-1][0])):
            c = pred_cls[-1][0][i].argmax()
            if c == 1:
                mask = model.sigmoid_img(pred_masks[-1][0][i]).unsqueeze(0) > 0.5
                all_masks_sig.append(mask)
        pred = functools.reduce(torch.bitwise_or, all_masks_sig).squeeze().T

        # Flip masks
        # pred = torch.flip(pred, dims=(0,))
        # instances_gt = torch.flip(instances_gt, dims=(0,))

        # Plot
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        axes[0].set_xlabel('y')
        axes[0].set_ylabel('x')
        axes[0].imshow(encoded_img)
        axes[1].imshow(pred)
        axes[2].imshow(instances_gt > 0)

        fig.tight_layout()
        fig.show()

    def test_weird_labels(self):
        pl.seed_everything(123)

        config_path = pathlib.Path('configs/old_training/semantic_kitti/21_point_mask_data_aug_gentle.yml')

        # Load model
        exp_name = config_path.stem
        with open(config_path, 'r') as f:
            config: dict = yaml.safe_load(f)
        config['shuffle_train'] = False
        config['batch_size'] = 1
        config['num_workers'] = 0

        datamodule = SemanticKittiMaskDataModule('data/SemanticKITTI', **config)
        gen_figs = True

        if gen_figs:
            dataloader = datamodule.val_dataloader()
            iterator = iter(dataloader)
            for j in range(5):
                # some good
                # for i in range(23):
                # some bad
                # for i in range(18):
                # for i in range(17):
                # for i in range(15):
                # for i in range(13):
                # for i in range(50):
                # for _ in range(4):
                point_clouds, (labels_gt, masks_gt), _ = next(iterator)
                instances_gt = torch.zeros((500, 500, 3))
                for i, m in enumerate(masks_gt[0]):
                    r = np.random.uniform(0, 1)
                    g = np.random.uniform(0, 1)
                    b = np.random.uniform(0, 1)
                    instances_gt[m > 0, :] = 1

                # Get mask and encoded point cloud
                point_clouds[0]
                gt_img = Image.fromarray(np.uint8(instances_gt[:, :] * 255))

                # Plot
                plt.imshow(instances_gt[:, :])
                plt.show()


if __name__ == '__main__':
    unittest.main()
