FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

# Set the current timezone
ENV TZ=America/Toronto \
    FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python and dependencies
RUN apt-get update && apt-get -y install software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y git ninja-build nvidia-cuda-toolkit libjpeg-dev zlib1g-dev libopenblas-dev
RUN apt-get install -y python3 python3-dev python3-venv python3-pip
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Preparing workdir
RUN mkdir /app
WORKDIR /app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==1.9.1 torchvision==0.10.1
RUN python3 -m pip install openmim

RUN mim install mmcv==2.0.0
RUN mim install mmdet==3.0.0
RUN mim install mmsegmentation==1.0.0
RUN mim install mmdet3d==1.1.0

ENV PYTHONPATH=/app:$PYTHONPATH
