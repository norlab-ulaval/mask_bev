FROM nvidia/cuda:11.8.0-devel-ubuntu18.04

# Set the current timezone
ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv A4B469963BF863CC
RUN gpg --export --armor A4B469963BF863CC | apt-key add -

RUN apt-get update && apt-get -y install software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y git ninja-build nvidia-cuda-toolkit python3.9 python3.9-dev python3.9-venv libjpeg-dev zlib1g-dev libopenblas-dev \
    && python3.9 -m ensurepip --default-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app
COPY ./scripts /app/scripts/
COPY requirements.txt /app/requirements.txt
COPY ./third_party /app/third_party/

# These commands needs the CUDA runtime
# RUN python3.9 -m pip install -r /app/requirements.txt
# RUN alias pip="python3.9 -m pip" && /app/scripts/install_deps.sh
# RUN python3.9 -m pip install networkx==2.8.7
# RUN python3.9 -m pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps

COPY . /app

