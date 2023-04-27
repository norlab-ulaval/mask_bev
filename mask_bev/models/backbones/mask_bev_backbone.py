from torch import nn

from mask_bev.models.networks.swin.swin import CustomSwinTransformer


# TODO add pre-trained backbone
class PointMaskBackbone(nn.Module):
    def __init__(self, pseudo_img_size: (int, int), in_channels: int, embded_dims: int, patch_size: int,
                 window_size: int, strides: (int, int, int, int), use_abs_enc: bool, swap_dims: bool = False,
                 backbone_overwrites=None):
        super().__init__()
        if backbone_overwrites is None:
            backbone_overwrites = dict()
        config = self._get_config(pseudo_img_size, in_channels, embded_dims, patch_size, window_size, strides,
                                  use_abs_enc, swap_dims)
        config.update(backbone_overwrites)
        self._backbone = CustomSwinTransformer(**config)
        self._backbone.init_weights()

    def forward(self, x):
        """
        Extract low-resolution features from a pseudo image
        :param x: tensor of shape (batch_size, channels_in, num_voxel_x, num_voxel_y)
        :return: list of tensor of shape (batch_size, channels_out, num_voxel_x/S_i, num_voxel_y/S_i) for some S_i
        """
        return self._backbone(x)

    def _resnet(self):
        return dict(
            depth=50,
            in_channels=32,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
        )

    def _get_config(self, pretrain_img_size: (int, int), in_channels: int, embed_dims: int, patch_size: int,
                    window_size: int, strides: (int, int, int, int), use_abs_pos_embed: bool, swap_dims: bool):
        return dict(
            pretrain_img_size=pretrain_img_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            patch_size=patch_size,
            window_size=window_size,
            mlp_ratio=4,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            strides=strides,
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.0,
            use_abs_pos_embed=use_abs_pos_embed,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            with_cp=False,
            init_cfg=None,
            swap_dims=swap_dims,
        )
