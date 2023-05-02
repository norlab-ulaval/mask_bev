# Training

Training parameters are provided via YAML configuration files. See `configs/jobs` for examples.
A guide on how to write your own configuration file can be found [here](CONFIGURATION.md).

## Local training

```shell
python train_mask_bev.py --config <path/to/config>
```

## Docker training

```shell
# Build docker image
docker build -t mask_bev .

# Run docker image
docker run --gpus all --rm -it -v <path/to/config>:/config -v <path/to/dataset>:/dataset -v <path/to/output>:/output openmim/maskbev:latest python train_mask_bev.py --config /config
```
