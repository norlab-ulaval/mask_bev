# Installation

Instructions for setting up a development environment for MaskBEV.
For Valeria specific instructions, see [Valeria instructions](docs/VALERIA.md).

Requires
- Python 3.10

```shell
pip install -r requirements.txt
# If you want to use visualization tools
pip install -r requirements-visualization.txt

# Install mmlabs packages
# Be sure to activate the venv again after install openmim
source venv/bin/activate
mim install mmcv
mim install mmdet
mim install mmsegmentation
mim install mmdet3d

mim install mmcv==2.0.0
mim install mmdet==3.0.0
mim install mmsegmentation==1.0.0
mim install mmdet3d==1.1.0
```
