#FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set the current timezone
ENV TZ=America/Toronto \
    FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python and dependencies
RUN apt-get update && apt-get -y install software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y git ninja-build nvidia-cuda-toolkit libjpeg-dev zlib1g-dev libopenblas-dev curl
RUN apt-get install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Preparing workdir
RUN mkdir /app
WORKDIR /app
RUN python3.10 -m pip install --upgrade pip
COPY requirements.txt /app
RUN python3.10 -m pip install -r requirements.txt

RUN mim install mmcv==2.0.0
RUN mim install mmdet==3.0.0
RUN mim install mmsegmentation==1.0.0
RUN mim install mmdet3d==1.1.0

ENV PYTHONPATH=/app:$PYTHONPATH
