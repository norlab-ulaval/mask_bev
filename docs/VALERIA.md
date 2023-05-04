# Valeria

## Datasets

Follow [`Valeria` from Norlab_wiki](https://github.com/norlab-ulaval/Norlab_wiki/wiki/Valeria) to upload the datasets to Valeria.

Also add symbolic links in `data/` to the datasets.

## Installation

```shell
# Load modules
module load python/3.10
module load cuda/11.7
module load qt
module load geos
module load llvm
module load gcc
module load opencv
module load scipy-stack
module load openblas

# Go to project root
cd mask_bev

# Create virtual environment
virtualenv --no-download venv
source venv/bin/activate
# or in tmp dir
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install requirements
pip install -U pip
pip install -r requirements-valeria.txt

# Install mmlabs packages
# Be sure to activate the venv again after install openmim
source venv/bin/activate
# or in tmp dir
source $SLURM_TMPDIR/venv/bin/activate

mim install mmcv==2.0.0
mim install mmdet==3.0.0
mim install mmsegmentation==1.0.0
mim install mmdet3d==1.1.0
```

Test it out

```shell
# Start an interactive session
salloc --time=60:00 --cpus-per-task=4 --mem=12G --partition=gpu --gres=gpu:a100:1
# TODO
```
