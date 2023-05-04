#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00
#SBATCH --job-name=generate_masks
#SBATCH --output=%x-%j.out

# Variables
# NAME: name of the job
# CONFIG: path to the config file

cd ~/mask_bev
docker build -t mask_bev .
docker run --gpus $CUDA_VISIBLE_DEVICES --rm \
  --mount type=bind,source="$(pwd)",target=/app/ \
  --mount type=bind,source="$(pwd)"/data/SemanticKITTI,target=/app/data/SemanticKITTI \
  --mount type=bind,source="$(pwd)"/data/KITTI,target=/app/data/KITTI \
  --mount type=bind,source="$(pwd)"/data/Waymo,target=/app/data/Waymo \
  mask_bev PYTHONPATH=. python3.10 scripts/generate_semantic_masks.py
