FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

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
COPY requirements.txt /app/requirements.txt
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install -r /app/requirements.txt
RUN mim install mmcv
RUN mim install mmdet
RUN mim install mmsegmentation
RUN mim install mmdet3d
ENV PYTHONPATH=/app:$PYTHONPATH
