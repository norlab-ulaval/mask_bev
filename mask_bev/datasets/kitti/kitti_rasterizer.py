import cv2
import numpy as np

from mask_bev.datasets.kitti.kitti_dataset import KittiFrame, KittiLabel, KittiType


class KittiRasterizer:
    def __init__(self, x_range: (int, int), y_range: (int, int), z_range: (int, int), voxel_size: float,
                 remove_unseen: bool = False, min_points: int = 1):
        """
        Converts a point cloud to an image where each instance has a different value
        :param x_range: x range of the image
        :param y_range: y range of the image
        :param z_range: z range of the image
        :param voxel_size: size of each pixel in the mask
        :param remove_unseen: remove instances not visible from the center scan in the mask
        :param min_points: minimum number of points to be considered seen
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
        self._accepted_types = {KittiType.Car}
        self._car_like_labels = {KittiType.Car, KittiType.Van, KittiType.Truck}
        self._label_mapping = {
            KittiType.Car: KittiType.Car,
            KittiType.Van: KittiType.Car,
            KittiType.Truck: KittiType.Car
        }

    def get_mask(self, frame: KittiFrame):
        out_voxels = dict()
        for t in self._accepted_types:
            out_voxels[t] = np.zeros((self._num_voxel_y, self._num_voxel_x), dtype=int)

        labels: [KittiLabel] = frame.labels
        labels: [KittiLabel] = [label for label in labels if label.type in self._car_like_labels]
        # TODO split by difficulty
        boxes: [(KittiLabel, KittiType)] = [(label, label.type) for label in labels]
        for instance, (b, t) in enumerate(boxes):
            if not (self._x_range[0] <= b.location[0] and self._x_range[1] and self._y_range[0] <= b.location[1] and
                    self._y_range[1]):
                continue
            b_contours = self._box_to_points(b)
            b_contours[:, 0] = self._map_to(b_contours[:, 0], self._x_range[0], self._x_range[1], 0, self._num_voxel_x)
            b_contours[:, 1] = self._map_to(b_contours[:, 1], self._y_range[0], self._y_range[1], 0, self._num_voxel_y)
            b_contours = np.intp(b_contours)
            t = self._label_mapping[t]
            mask = np.zeros_like(out_voxels[t], dtype=np.uint8)
            cv2.drawContours(mask, [b_contours], 0, 255, -1)
            out_voxels[t][mask > 0] = instance + 1

        return out_voxels

    @staticmethod
    def _map_to(value, imin, imax, omin, omax):
        return (value - imin) / (imax - imin) * (omax - omin) + omin

    def _box_to_points(self, b: KittiLabel):
        points = np.zeros((4, 2))
        cx, cy, cz, l, w, h, theta = *b.location, *b.dimensions, b.rotation_y
        dl, dw = l / 2, w / 2
        center = [cx, cy]
        d = np.array([np.cos(theta), np.sin(theta)])
        phi = theta + np.pi / 2
        d_bar = np.array([np.cos(phi), np.sin(phi)])

        points[0, :] = d * dl + d_bar * dw
        points[1, :] = - d * dl + d_bar * dw
        points[2, :] = - d * dl - d_bar * dw
        points[3, :] = d * dl - d_bar * dw

        points += center

        return points
