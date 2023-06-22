import pickle

import cv2
import numpy as np
import tqdm

from mask_bev.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset, \
    SemanticKittiRawLabel, SemanticKittiScan
from mask_bev.datasets.semantic_kitti.semantic_kitti_scene import SceneMaker, SemanticKittiScene


class SemanticKittiRasterizer:
    def __init__(self, x_range: (int, int), y_range: (int, int), z_range: (int, int), voxel_size: float,
                 remove_unseen: bool = False, min_points: int = 1, morph_kernel_size: int = 9):
        """
        Converts a point cloud to an image where each instance has a different value
        :param x_range: x range of the image
        :param y_range: y range of the image
        :param z_range: z range of the image
        :param voxel_size: size of each pixel in the mask
        :param remove_unseen: remove instances not visible from the center scan in the mask
        :param min_points: minimum number of points to be considered seen
        :param morph_kernel_size: size of the kernel used for morphological operations
        """
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range
        self._voxel_size = voxel_size
        self._num_voxel_x = int((x_range[1] - x_range[0]) / voxel_size)
        self._num_voxel_y = int((y_range[1] - y_range[0]) / voxel_size)
        self._num_classes = self._num_voxel_x * self._num_voxel_y
        self._remove_unseen = remove_unseen
        self._min_points = min_points
        self._morph_kernel_size = morph_kernel_size

        self._lattice_idx = np.mgrid[0:self._num_voxel_x, 0:self._num_voxel_y]
        self._lattice_lower_left_corner = (self._lattice_idx.T * voxel_size + [self._x_range[0], self._y_range[0]]).T
        self._lattice_center = (self._lattice_lower_left_corner + voxel_size / 2)
        self._idx = None

    def get_mask_around(self, scan: SemanticKittiScan, scene: SemanticKittiScene):
        """
        Generates a mask where each pixel corresponds to what instance is present in the corresponding voxel
        :param scan: center scan around which to generate the mask
        :param scene: scene containing all points of all the scans in a sequence
        :param filter_bad_instances: whether to filter out instances with less than min_inst_pixel pixels
        :return: mask of the shape (num_voxel_x, num_voxel_y)
        """
        scene_pc = scene.point_cloud
        scene_inst = scene.inst_label

        # make scene relative to scan
        scene_pc_homo = np.copy(scene_pc)
        scene_pc_homo[:, 3] = 1
        scene_pc_homo = (scan.velo_to_inv_pose @ scene_pc_homo.T).T
        scene_pc_homo /= scene_pc_homo[:, 3].reshape((-1, 1))

        # only keep points in range
        in_range = (self._x_range[0] < scene_pc_homo[:, 0]) & (scene_pc_homo[:, 0] < self._x_range[1]) & \
                   (self._y_range[0] < scene_pc_homo[:, 1]) & (scene_pc_homo[:, 1] < self._y_range[1]) & \
                   (self._z_range[0] < scene_pc_homo[:, 2]) & (scene_pc_homo[:, 2] < self._z_range[1])
        scene_pc_homo = scene_pc_homo[in_range]
        scene_inst = scene_inst[in_range]

        # find corresponding voxel of each point
        pc_x_idx = ((scene_pc_homo[:, 0] - self._x_range[0]) // self._voxel_size).astype(int)
        pc_y_idx = ((scene_pc_homo[:, 1] - self._y_range[0]) // self._voxel_size).astype(int)
        idx = np.stack([pc_x_idx, pc_y_idx]).T

        # generate mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self._morph_kernel_size, self._morph_kernel_size))
        out_voxels = np.zeros((self._num_voxel_x, self._num_voxel_y), dtype=int)
        if self._remove_unseen:
            instances_in_scan = set(scan.inst_label) - {0}
            present_instances = set()
            for i in instances_in_scan:
                if np.count_nonzero(scan.inst_label == i) >= self._min_points:
                    present_instances.add(i)
        else:
            present_instances = set(scene_inst) - {0}

        for instance in present_instances:
            instance_voxel = np.zeros_like(out_voxels, dtype=np.uint8)
            instance_idx = idx[scene_inst == instance]
            instance_voxel[instance_idx[:, 0], instance_idx[:, 1]] = 1

            instance_voxel = cv2.morphologyEx(instance_voxel, cv2.MORPH_CLOSE, kernel)
            instance_voxel = cv2.morphologyEx(instance_voxel, cv2.MORPH_OPEN, kernel)

            instance_mask = instance_voxel > 0.5

            out_voxels[instance_mask] = instance
        self._idx = idx
        return out_voxels


if __name__ == '__main__':
    root = 'data/SemanticKITTI'
    train_dataset = SemanticKittiSequenceDataset(root, 'train', included_labels=SemanticKittiRawLabel.CAR)
    areas = []

    for sequence in train_dataset:
        scan_idx = sequence.scan_indices
        for i, scan in tqdm.tqdm(enumerate(train_dataset.load_scan_indices(scan_idx)), total=len(scan_idx)):
            max_points = sum(s.num_points for s in train_dataset.load_scan_indices([scan_idx[i]]))
            scene_maker = SceneMaker(max_points=max_points)
            scene_maker.add_scan(scan)
            scene = scene_maker.scene
            rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.16)
            mask = rasterizer.get_mask_around(scan, scene)
            instances = set(np.unique(mask)) - {0}
            for inst in instances:
                areas.append((mask == inst).sum())
    with open('data/SemanticKITTI/mask_area_no_completion.pkl', 'wb') as f:
        pickle.dump(areas, f)
    exit(0)

    sequence = train_dataset[0]
    num_instances = 0
    for sequence in train_dataset:
        scan_idx = sequence.scan_indices[:500:2]
        max_points = sum(s.num_points for s in train_dataset.load_scan_indices(scan_idx))
        scene_maker = SceneMaker(max_points=max_points)
        for scan in tqdm.tqdm(train_dataset.load_scan_indices(scan_idx), total=len(scan_idx)):
            scene_maker.add_scan(scan)

        scan = train_dataset.load_scan_index(scan_idx[len(scan_idx) // 2])
        scene = scene_maker.scene
        scene.point_cloud = scene.point_cloud
        scene.sem_label = scene.sem_label
        scene.inst_label = scene.inst_label

        rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.16)
        mask = rasterizer.get_mask_around(scan, scene)
        print(mask.shape)

        instances = set(np.unique(scene.inst_label)) - {0}

        num_instances += len(instances)

    print(f'{num_instances=}')
    plt.imshow(mask)
    plt.colorbar()
    plt.show()
