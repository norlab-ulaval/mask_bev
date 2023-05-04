#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=4-00:00
#SBATCH --job-name=train_mask_bev
#SBATCH --output=%x-%j.out

# Load modules
module load python/3.9
module load cuda/11.7
module load qt
module load geos
module load llvm
module load gcc
module load opencv
module load scipy-stack
module load openblas

# Start training
cd ~/mask_bev
source venv/bin/activate
python train_mask_bev.py --config configs/training/semantic_kitti/00_quick_test.yml
