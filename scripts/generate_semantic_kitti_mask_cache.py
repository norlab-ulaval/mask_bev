import multiprocessing

import numpy as np
import tqdm

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset, SemanticKittiRawLabel
from mask_bev.datasets.semantic_kitti.semantic_kitti_mask_dataset import SemanticKittiMaskDataset

if __name__ == '__main__':
    root = 'data/SemanticKITTI'
    train_dataset = SemanticKittiSequenceDataset(root, 'train', included_labels=SemanticKittiRawLabel.CAR)
    sequence = train_dataset[0]

    x_range, y_range, z_range, voxel_size = (-40, 40), (-40, 40), (-10, 10), 0.16


    def gen_mask(i):
        mask_dataset = SemanticKittiMaskDataset(train_dataset, x_range, y_range, z_range, voxel_size,
                                                remove_unseen=True, min_points=1)
        m = mask_dataset[i]
        return len(np.unique(m.mask))


    mask_dataset = SemanticKittiMaskDataset(train_dataset, x_range, y_range, z_range, voxel_size, remove_unseen=True,
                                            min_points=1)
    num_masks = len(mask_dataset)
    with multiprocessing.Pool(32, maxtasksperchild=1) as p:
        iterator = tqdm.tqdm(p.imap_unordered(gen_mask, range(num_masks)), total=num_masks)
        print(f'Max number of instance per scan {max(iterator)}')
