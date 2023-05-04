#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00
#SBATCH --job-name=generate_masks
#SBATCH --output=%x-%j.out

cd ~/mask_bev
docker build -t mask_bev .
docker run --gpus $CUDA_VISIBLE_DEVICES --rm \
  --mount type=bind,source="$(pwd)",target=/app/ \
  --mount type=bind,source="$(pwd)"/data/SemanticKITTI,target=/app/data/SemanticKITTI \
  --mount type=bind,source="$(pwd)"/data/KITTI,target=/app/data/KITTI \
  --mount type=bind,source="$(pwd)"/data/Waymo,target=/app/data/Waymo \
  mask_bev python3.10 train_mask_bev.py --config $CONFIG
