#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-04:00
#SBATCH --partition=gpu
#SBATCH --job-name=generate_masks
#SBATCH --output=%x-%j.out

cd ~/mask_bev
source venv/bin/activate
PYTHONPATH="${PYTHONPATH}:."  python train_mask_bev.py --config $CONFIG
