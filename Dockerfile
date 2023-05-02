FROM nvidia/cuda:11.8.0-devel-ubuntu18.04

# Set the current timezone
ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.9 and dependencies
RUN gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv A4B469963BF863CC
RUN gpg --export --armor A4B469963BF863CC | apt-key add -
RUN apt-get update && apt-get -y install software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y git ninja-build nvidia-cuda-toolkit python3.10 python3.10-dev python3.10-venv libjpeg-dev zlib1g-dev libopenblas-dev \
    && python3.10 -m ensurepip --default-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

# Copying the code
COPY . /app
