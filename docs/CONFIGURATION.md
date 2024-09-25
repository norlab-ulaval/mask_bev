# Parameters

## General

`seed`: random seed for reproducibility.

## Model

`checkpoint`: path to a checkpoint to load.

`lr`: learning rate.

`weight_decay`: weight decay.

`optimizer_type`: optimizer type. Choose from `adam`, `lamb`, `sgd`, `adam_w`.

`lr_schedulers_type`: learning rate scheduler type. Choose from `step`, `plateau`, `cosine`, `poly`.

`differential_lr`: Whether to apply a learning rate multiplier to the backbone.

`differential_lr_scaling`: LR multiplier for backbone.

`x_range`, `y_range`, `z_range`: Range of the input point cloud.

`voxel_size`: Voxel size of the input point cloud.

`num_queries`: Number of queries for Mask2Former. Should be higher than the number of objects per scan.

`predict_height`: Whether to predict the height of the objects.

`pc_point_dim`: Number of dimensions of the input point cloud.

## Encoder

`max_num_points`: Maximum number of points per voxel to keep at the encoder stage.

`encoder_feat_channels`: Number of channels of the encoder features.

`encoder_encoding_type`: Encoding type of the encoder. Choose from `vanilla`, `fourier`, `cosine`.

`encoder_fourier_enc_group`: Number of groups for the Fourier encoding.

## Backbone

`backbone_embed_dim`: Embedding dimension of the backbone.

`backbone_use_abs_emb`: Whether to use absolute positional embeddings in the backbone.

`backbone_path_size`: Patch size of the Swin Transformer backbone.

`backbone_window_size`: Window size of the Swin Transformer backbone.

`backbone_strides`: Strides of the Swin Transformer backbone (int, int, int, int).

`backbone_swap_dims`: Whether to swap the dimensions of the Swin Transformer backbone.

## Head

`head_feat_channels`: Number of channels of the head features.

`head_out_channels`: Number of channels of the head output.

`head_reverse_class_weight`: Whether to reverse the class weight. This changes the default Mask2Former weighting of the
background class.

`head_num_classes`: Number of classes of the head output.

## Dataset

`min_num_inst_pixels`: Minimum number of pixels to keep an instance in the dataset.

`batch_size`: Batch size.

`test_batch_size`: Batch size for testing.

`num_workers`: Number of workers for the data loader.

`test_num_workers`: Number of workers for the data loader for testing.

`pin_memory`: Whether to pin memory for the data loader.

`remove_unseen`: Whether to remove instances that do not appear in a scan.

`shuffle_train`: Whether to shuffle the training data.

`min_num_points`: Minimum number of points to keep an instance in a scan.

`log_every_n_step`: Logging frequency during training.

`limit_train_batches`: Maximum number of batches to train on.

`limit_val_batches`: Maximum number of batches to validate on.

### Augmentations

Available augmentations are available in `maskbev.augmentations`.
Select the ones you want using

```yaml
augmentations:
  - name: <augmentation name>
      <augmentation parameters>
  - name: <augmentation name>
      <augmentation parameters>
```

