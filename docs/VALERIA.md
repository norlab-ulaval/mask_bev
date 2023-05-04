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

# Install requirements
pip install -U pip
pip install -r requirements_valeria.txt
# Install mmlabs packages
# Be sure to activate the venv again after install openmim
source venv/bin/activate
mim install mmcv
mim install mmdet
mim install mmsegmentation
mim install mmdet3d
```

Test it out

```shell
# Start an interactive session
salloc --account=ul-val-prj-def-phgig4 --cpus-per-task=8 --time=2:00:00 --gres=gpu:a100:1 --partition=gpu --mem=12G
# TODO
```
