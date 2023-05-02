from typing import Union, Dict, Optional

import einops as ein
import torch
from mmcv.ops import Voxelization
from mmdet3d.models import PillarFeatureNet, PointPillarsScatter
from torch import nn
from torch.nn import functional as F

from mask_bev.models.positional_encoding.learnable_fourier_positional_encoding import \
    LearnableFourierPositionalEncoding


class EncodingType:
    Vanilla = 'vanilla'
    Fourier = 'fourier'


class MaskBevEncoder(nn.Module):
    def __init__(self, feat_channels: [int], x_range: (int, int), y_range: (int, int),
                 z_range: (int, int), voxel_size_x: float, voxel_size_y: float, voxel_size_z: float,
                 max_num_points: int, encoding_type: str, fourier_enc_group: int,
                 max_voxels: Union[tuple, int] = 500 * 500, deterministic: bool = True,
                 encoder_params: Optional[Dict] = None, pc_point_dim: int = 4):
        """
        Encode point clouds into a pseudo image
        :param feat_channels: number of channels in the FPN, that last one in the list is the output dimension
        :param x_range: x range (min, max)
        :param y_range: y range (min, max)
        :param z_range: z range (min, max)
        :param voxel_size_x: voxel size in x
        :param voxel_size_y: voxel size in y
        :param voxel_size_z: voxel size in z
        :param max_num_points: max number of points per voxel
        :param encoding_type: Type of encoder to use, check `EncodingType` for supported encoders
        :param max_voxels: max number of voxel (training, testing)
        :param deterministic: non-deterministic is faster, but less stable
        :param encoder_params: params to pass directly to `PillarFeatureNet`
        """
        super().__init__()
        if encoder_params is None:
            encoder_params = {}
        self._feat_channels = feat_channels
        self._out_features = feat_channels[-1]
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range

        if encoding_type == EncodingType.Vanilla:
            self._pos_encoder = None
            pc_in_channels = pc_point_dim
        elif encoding_type == EncodingType.Fourier:
            pc_in_channels = 128
            self._pos_encoder_group = fourier_enc_group
            self._pos_encoder_M = 4 // self._pos_encoder_group
            self._pos_encoder = LearnableFourierPositionalEncoding(G=self._pos_encoder_group, M=self._pos_encoder_M,
                                                                   F_dim=32, H_dim=32, D=pc_in_channels, gamma=1.0)
        else:
            raise NotImplementedError(f'{encoding_type}')

        self._num_voxel_x = int((x_range[1] - x_range[0]) / voxel_size_x)
        self._num_voxel_y = int((y_range[1] - y_range[0]) / voxel_size_y)
        self._num_voxel_z = 1

        point_cloud_range = [x_range[0], y_range[0], z_range[0], x_range[1], y_range[1], z_range[1]]
        voxel_size = [voxel_size_x, voxel_size_y, voxel_size_z]
        self._voxel_layer = Voxelization(voxel_size, point_cloud_range, max_num_points, max_voxels, deterministic)
        self._voxel_encoder = PillarFeatureNet(in_channels=pc_in_channels, feat_channels=self._feat_channels,
                                               voxel_size=voxel_size,
                                               point_cloud_range=point_cloud_range, **encoder_params)
        out_shape = [self._num_voxel_y, self._num_voxel_x]
        self._middle_encoder = PointPillarsScatter(in_channels=self._out_features, output_shape=out_shape)
        self._layer_norm = nn.LayerNorm([self._out_features, *out_shape], eps=1e-3)

    def forward(self, point_clouds):
        """
        Encodes a point cloud into a pseudo image
        :param point_clouds: list of tensors [Tensor(Ni, 4)], len(point_clouds) = batch_size
        :return: tensor of shape (num_voxels, feat_channels[-1])
        """
        batch_size = len(point_clouds)
        voxels, num_points, coors = self.voxelize(point_clouds)
        if self._pos_encoder is not None:
            v, num_pts, _ = voxels.shape
            x = ein.rearrange(voxels, 'v n c -> (v n) c').reshape((-1, self._pos_encoder_group, self._pos_encoder_M))
            x = self._pos_encoder(x)
            voxels = ein.rearrange(x, '(v n) c -> v n c', v=v, n=num_pts)
        voxel_features = self.encode(voxels, num_points, coors)
        pseudo_img = self.middle_encode(voxel_features, coors, batch_size)
        pseudo_img = self._layer_norm(pseudo_img)
        return pseudo_img

    def voxelize(self, point_clouds):
        # See VoxelNet from mmdetection3d
        voxels, coors, num_points = [], [], []
        for res in point_clouds:
            res = self._filter_in_range(res)
            res_voxels, res_coors, res_num_points = self._voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def _filter_in_range(self, point_cloud):
        in_range = (self._x_range[0] < point_cloud[:, 0]) & (point_cloud[:, 0] < self._x_range[1]) & \
                   (self._y_range[0] < point_cloud[:, 1]) & (point_cloud[:, 1] < self._y_range[1]) & \
                   (self._z_range[0] < point_cloud[:, 2]) & (point_cloud[:, 2] < self._z_range[1])
        return point_cloud[in_range]

    def encode(self, voxel, num_points, coords):
        return self._voxel_encoder(voxel, num_points, coords)

    def middle_encode(self, voxel_features, coors, batch_size=None):
        return self._middle_encoder(voxel_features, coors, batch_size)

# TODO implement other encoders
# class CustomVoxelEncoder(nn.Module, ABC):
#     """Cross Attention Voxel Encoder
#
#     Uses attention to generate features from dynamic voxels.
#     Args:
#         in_channels (int, optional): Number of input features,
#             either x, y, z or x, y, z, r. Defaults to 4.
#         feat_channels (tuple, optional): Number of features in each of the
#             N PFNLayers. Defaults to (64, ).
#         with_distance (bool, optional): Whether to include Euclidean distance
#             to points. Defaults to False.
#         with_cluster_center (bool, optional): [description]. Defaults to True.
#         with_voxel_center (bool, optional): [description]. Defaults to True.
#         voxel_size (tuple[float], optional): Size of voxels, only utilize x
#             and y size. Defaults to None.
#         point_cloud_range (tuple[float], optional): Point cloud range, only
#             utilizes x and y min. Defaults to None.
#     """
#
#     def __init__(self,
#                  in_channels=4,
#                  feat_channels=(64,),
#                  with_distance=False,
#                  with_cluster_center=True,
#                  with_voxel_center=True,
#                  voxel_size=None,
#                  point_cloud_range=None,
#                  norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
#                  mode='max',
#                  legacy=True):
#         super().__init__()
#         assert len(feat_channels) > 0
#         assert voxel_size is not None
#         assert point_cloud_range is not None
#
#         self.legacy = legacy
#
#         if with_cluster_center:
#             in_channels += 3
#         if with_voxel_center:
#             in_channels += 2
#         if with_distance:
#             in_channels += 1
#
#         self.in_channels = in_channels
#         self.feat_channels = feat_channels
#         self.with_distance = with_distance
#         self.with_cluster_center = with_cluster_center
#         self.with_voxel_center = with_voxel_center
#         self.voxel_size = voxel_size
#         self.point_cloud_range = point_cloud_range
#         self.fp16_enabled = False
#
#         self.vx = voxel_size[0]
#         self.vy = voxel_size[1]
#         self.vz = point_cloud_range[2] - point_cloud_range[5]
#         self.x_offset = self.vx / 2 + point_cloud_range[0]
#         self.y_offset = self.vy / 2 + point_cloud_range[1]
#         self.point_cloud_range = point_cloud_range
#
#     @abstractmethod
#     def voxel_encoding(self, features, num_points, coors):
#         ...
#
#     def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
#         """Map the centers of voxels to its corresponding points.
#
#         Args:
#             pts_coors (torch.Tensor): The coordinates of each points, shape
#                 (M, 3), where M is the number of points.
#             voxel_mean (torch.Tensor): The mean or aggreagated features of a
#                 voxel, shape (N, C), where N is the number of voxels.
#             voxel_coors (torch.Tensor): The coordinates of each voxel.
#
#         Returns:
#             torch.Tensor: Corresponding voxel centers of each points, shape
#                 (M, C), where M is the numver of points.
#         """
#         # Step 1: scatter voxel into canvas
#         # Calculate necessary things for canvas creation
#         canvas_y = int(
#             (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
#         canvas_x = int(
#             (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
#         canvas_channel = voxel_mean.size(1)
#         batch_size = pts_coors[-1, 0] + 1
#         canvas_len = canvas_y * canvas_x * batch_size
#         # Create the canvas for this sample
#         canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
#         # Only include non-empty pillars
#         indices = (
#                 voxel_coors[:, 0] * canvas_y * canvas_x +
#                 voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
#         # Scatter the blob back to the canvas
#         canvas[:, indices.long()] = voxel_mean.t()
#
#         # Step 2: get voxel mean for each point
#         voxel_index = (
#                 pts_coors[:, 0] * canvas_y * canvas_x +
#                 pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
#         center_per_point = canvas[:, voxel_index.long()].t()
#         return center_per_point
#
#     def map_to(self, x, a, b, c, d):
#         return (x - a) / (b - a) * (d - c) + c
#
#     def linear_pos_encoding(self, coors, features, num_points):
#         dtype = features.dtype
#         # mean position in a voxel
#         num_pt_per_voxel = features.shape[1]
#         points_mean = features[:, :, :3].sum(
#             dim=1, keepdim=True) / num_points.type_as(features).view(
#             -1, 1, 1)
#         points_mean[:, :, 0] = self.map_to(points_mean[:, :, 0], self.point_cloud_range[0], self.point_cloud_range[3],
#                                            -1, 1)
#         points_mean[:, :, 1] = self.map_to(points_mean[:, :, 1], self.point_cloud_range[1], self.point_cloud_range[4],
#                                            -1, 1)
#         points_mean[:, :, 2] = self.map_to(points_mean[:, :, 2], self.point_cloud_range[2], self.point_cloud_range[5],
#                                            -1, 1)
#         points_mean = ein.repeat(points_mean, 'n 1 c -> n p c', p=num_pt_per_voxel)
#         # local coords in a voxel
#         normalized = torch.zeros_like(features[:, :, :3])
#         normalized[:, :, 0] = (features[:, :, 0] - (
#                 coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)) / (self.vx / 2)
#         normalized[:, :, 1] = (features[:, :, 1] - (
#                 coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)) / (self.vy / 2)
#         normalized[:, :, 2] = self.map_to(features[:, :, 2],
#                                           self.point_cloud_range[2], self.point_cloud_range[5],
#                                           -1, 1)
#         # voxel cors
#         coors_pos = torch.zeros_like(features[:, :, :2])
#         coors_pos[:, :, 0] = self.map_to(coors[:, 3].to(dtype).unsqueeze(1) * self.vy + self.x_offset,
#                                          self.point_cloud_range[0], self.point_cloud_range[3], -1, 1)
#         coors_pos[:, :, 1] = self.map_to(coors[:, 1].to(dtype).unsqueeze(1) * self.vy + self.x_offset,
#                                          self.point_cloud_range[1], self.point_cloud_range[4], -1, 1)
#         # normalize features
#         features[:, :, 0] = self.map_to(features[:, :, 0], self.point_cloud_range[0], self.point_cloud_range[3], -1, 1)
#         features[:, :, 1] = self.map_to(features[:, :, 1], self.point_cloud_range[1], self.point_cloud_range[4], -1, 1)
#         features[:, :, 2] = self.map_to(features[:, :, 2], self.point_cloud_range[2], self.point_cloud_range[5], -1, 1)
#         features[:, :, 3] = self.map_to(features[:, :, 3], 0, 1, -1, 1)
#         features = torch.cat([features, points_mean, normalized, coors_pos], dim=-1)
#         return features
#
#     @force_fp32(apply_to=['features'], out_fp16=True)
#     def forward(self, features, num_points, coors):
#         features_ls = [features]
#         # Find distance of x, y, and z from cluster center
#         if self.with_cluster_center:
#             points_mean = features[:, :, :3].sum(
#                 dim=1, keepdim=True) / num_points.type_as(features).view(
#                 -1, 1, 1)
#             f_cluster = features[:, :, :3] - points_mean
#             features_ls.append(f_cluster)
#
#         # Find distance of x, y, and z from pillar center
#         dtype = features.dtype
#         if self.with_voxel_center:
#             if not self.legacy:
#                 f_center = torch.zeros_like(features[:, :, :2])
#                 f_center[:, :, 0] = features[:, :, 0] - (
#                         coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
#                         self.x_offset)
#                 f_center[:, :, 1] = features[:, :, 1] - (
#                         coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
#                         self.y_offset)
#             else:
#                 f_center = features[:, :, :2]
#                 f_center[:, :, 0] = f_center[:, :, 0] - (
#                         coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
#                         self.x_offset)
#                 f_center[:, :, 1] = f_center[:, :, 1] - (
#                         coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
#                         self.y_offset)
#             features_ls.append(f_center)
#
#         if self.with_distance:
#             points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
#             features_ls.append(points_dist)
#
#         # Combine together feature decorations
#         features = torch.cat(features_ls, dim=-1)
#         # The feature decorations were calculated without regard to whether
#         # pillar was empty. Need to ensure that
#         # empty pillars remain set to zeros.
#         voxel_count = features.shape[1]
#         mask = get_paddings_indicator(num_points, voxel_count, axis=0)
#         mask = torch.unsqueeze(mask, -1).type_as(features)
#         features *= mask
#
#         features = self.voxel_encoding(features, num_points, coors)
#
#         return features
#
#
# @VOXEL_ENCODERS.register_module()
# class CrossAttentionEncoder(CustomVoxelEncoder):
#     def __init__(self,
#                  in_channels=12,
#                  feat_channels=(64,),
#                  with_distance=False,
#                  with_cluster_center=False,
#                  with_voxel_center=False,
#                  voxel_size=None,
#                  point_cloud_range=None,
#                  norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
#                  mode='max',
#                  legacy=True,
#                  depth=1,
#                  dim=None,
#                  num_queries=1,
#                  queries_dim=64,
#                  num_latents=16,
#                  latent_dim=64,
#                  cross_heads=2,
#                  latent_heads=2,
#                  cross_dim_head=32,
#                  latent_dim_head=32,
#                  weight_tie_layers=False,
#                  decoder_ff=True):
#         super(CrossAttentionEncoder, self).__init__(
#             in_channels=in_channels,
#             feat_channels=feat_channels,
#             with_distance=with_distance,
#             with_cluster_center=with_cluster_center,
#             with_voxel_center=with_voxel_center,
#             voxel_size=voxel_size,
#             point_cloud_range=point_cloud_range,
#             norm_cfg=norm_cfg,
#             mode=mode,
#             legacy=legacy,
#         )
#
#         self.queries = nn.Parameter(torch.randn(num_queries, queries_dim))
#         self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
#
#         if dim is not None:
#             print('`dim` is deprecated, it is computed automatically')
#         dim = self.in_channels
#         self.cross_attend_blocks = nn.ModuleList([
#             pio.PreNorm(latent_dim, pio.Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head),
#                         context_dim=dim),
#             pio.PreNorm(latent_dim, pio.FeedForward(latent_dim))
#         ])
#
#         get_latent_attn = lambda: pio.PreNorm(latent_dim,
#                                               pio.Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
#         get_latent_ff = lambda: pio.PreNorm(latent_dim, pio.FeedForward(latent_dim))
#         get_latent_attn, get_latent_ff = map(pio.cache_fn, (get_latent_attn, get_latent_ff))
#
#         self.layers = nn.ModuleList([])
#         cache_args = {'_cache': weight_tie_layers}
#
#         for i in range(depth):
#             self.layers.append(nn.ModuleList([
#                 get_latent_attn(**cache_args),
#                 get_latent_ff(**cache_args)
#             ]))
#
#         self.decoder_cross_attn = pio.PreNorm(queries_dim, pio.Attention(queries_dim, latent_dim, heads=cross_heads,
#                                                                          dim_head=cross_dim_head),
#                                               context_dim=latent_dim)
#         self.decoder_ff = pio.PreNorm(queries_dim, pio.FeedForward(queries_dim)) if decoder_ff else None
#
#     def voxel_encoding(self, features, num_points, coors):
#         features = self.linear_pos_encoding(coors, features, num_points)
#
#         voxel_count = features.shape[1]
#         mask = get_paddings_indicator(num_points, voxel_count, axis=0)
#         mask = torch.unsqueeze(mask, -1).type_as(features)
#         features *= mask
#
#         nb_voxels, *_, device = *features.shape, features.device
#         x = repeat(self.latents, 'n d -> nb_voxels n d', nb_voxels=nb_voxels)
#         cross_attn, cross_ff = self.cross_attend_blocks
#         x = cross_attn(x, context=features, mask=None) + x
#         x = cross_ff(x) + x
#         for self_attn, self_ff in self.layers:
#             x = self_attn(x) + x
#             x = self_ff(x) + x
#
#         # FTODO generate queries from features
#         queries = self.queries
#         if not pio.exists(self.queries):
#             return x
#
#         # make sure queries contains batch dimension
#         if queries.ndim == 2:
#             queries = repeat(queries, 'n d -> nb_voxels n d', nb_voxels=nb_voxels)
#
#         # cross attend from decoder queries to latents
#         latents = self.decoder_cross_attn(queries, context=x)[:, 0, :].squeeze()
#
#         # optional decoder feedforward
#         if pio.exists(self.decoder_ff):
#             latents = latents + self.decoder_ff(latents)
#
#         return latents.squeeze()
#
#
# @VOXEL_ENCODERS.register_module()
# class DGCNNEncoder(CustomVoxelEncoder):
#     def __init__(self,
#                  in_channels=4,
#                  feat_channels=(64,),
#                  with_distance=False,
#                  with_cluster_center=False,
#                  with_voxel_center=False,
#                  voxel_size=None,
#                  point_cloud_range=None,
#                  norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
#                  mode='max',
#                  legacy=True,
#                  dim=12,
#                  k=8,
#                  aggr='max',
#                  intermediate_features: Optional[list] = None,
#                  encoding_size=64,
#                  per_voxel=True,
#                  num_workers=1):
#         super(DGCNNEncoder, self).__init__(
#             in_channels=in_channels,
#             feat_channels=feat_channels,
#             with_distance=with_distance,
#             with_cluster_center=with_cluster_center,
#             with_voxel_center=with_voxel_center,
#             voxel_size=voxel_size,
#             point_cloud_range=point_cloud_range,
#             norm_cfg=norm_cfg,
#             mode=mode,
#             legacy=legacy,
#         )
#
#         if intermediate_features is None:
#             intermediate_features = []
#
#         layers = []
#         intermediate_features = [dim] + intermediate_features + [encoding_size]
#
#         for feat_in, feat_out in zip(intermediate_features, intermediate_features[1:]):
#             dgcnn_layers = self.make_edge_conv(feat_in, feat_out, k, aggr, num_workers)
#             layers.append(dgcnn_layers)
#
#         self.edge_convs = nn.ModuleList(layers)
#         self.per_voxel = per_voxel
#
#     def make_edge_conv(self, in_channels, out_channels, k, aggr, num_workers):
#         class HTheta(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.linear = nn.Linear(2 * in_channels, out_channels)
#
#             def forward(self, x):
#                 x = torch.cat([x[:, :in_channels], x[:, in_channels:]], dim=1)
#                 return self.linear(x)
#
#         return gnn.DynamicEdgeConv(HTheta(), k=k, aggr=aggr, num_workers=num_workers)
#
#     def voxel_encoding(self, features, num_points, coors):
#         features = self.linear_pos_encoding(coors, features, num_points)
#
#         (num_voxels, pts_per_voxel, dim), device = features.shape, features.device
#         features = ein.rearrange(features, 'num_voxel num_pts dim -> (num_voxel num_pts) dim')
#         # FTODO per voxel only
#         if self.per_voxel:
#             batch = torch.arange(num_voxels, device=device)
#             batch = ein.repeat(batch, 'num_voxels -> (num_voxels pts_per_voxel)', pts_per_voxel=pts_per_voxel)
#         else:
#             batch = ein.repeat(coors[:, 0].long(), 's -> s pts_per_voxel', pts_per_voxel=pts_per_voxel)
#             batch = ein.rearrange(batch, 's p -> (s p)')
#
#         for edge_conv in self.edge_convs:
#             features = edge_conv(features, batch)
#
#         features = ein.rearrange(features, '(num_voxels pts_per_voxel) d -> num_voxels pts_per_voxel d',
#                                  num_voxels=num_voxels, pts_per_voxel=pts_per_voxel)
#         features = ein.rearrange(features, 'num_voxel num_pts dim -> num_voxel dim num_pts')
#         features = F.max_pool1d(features, 32)
#         return features.squeeze()
#
#
# @VOXEL_ENCODERS.register_module()
# class FKAConvEncoder(CustomVoxelEncoder):
#     def __init__(self,
#                  in_channels=4,
#                  feat_channels=(64,),
#                  voxel_size=None,
#                  point_cloud_range=None,
#                  norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
#                  mode='max',
#                  legacy=True,
#                  encoding_size=64,
#                  kernel_size=16,
#                  intermediate_features: Optional[list] = None,
#                  bias=True):
#         super(FKAConvEncoder, self).__init__(
#             in_channels=in_channels,
#             feat_channels=feat_channels,
#             with_distance=False,
#             with_cluster_center=False,
#             with_voxel_center=False,
#             voxel_size=voxel_size,
#             point_cloud_range=point_cloud_range,
#             norm_cfg=norm_cfg,
#             mode=mode,
#             legacy=legacy,
#         )
#         # FTODO multiple layers with batchnorm in between? or layer norm?
#         self.fkaconv = FKAConv(in_channels=1, out_channels=encoding_size, kernel_size=kernel_size, bias=bias, dim=3)
#
#     def voxel_encoding(self, x, num_points, coors):
#         features = ein.rearrange(x, 'bs n_pts dim -> bs dim n_pts')
#         points = ein.rearrange(x[:, :, :3], 'b points dim -> b dim points')
#
#         support_points = torch.zeros_like(points)
#         support_points[:, :, 0] = coors[:, 3].type_as(x).unsqueeze(1) * self.vx + self.x_offset
#         support_points[:, :, 1] = coors[:, 2].type_as(x).unsqueeze(1) * self.vy + self.y_offset
#         support_points[:, :, 2] = (self.point_cloud_range[2] + self.point_cloud_range[5]) / 2
#
#         x = self.fkaconv(features, points, support_points)[0]
#
#         return torch.max(x.squeeze(), dim=2)[0]
