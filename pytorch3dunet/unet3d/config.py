import argparse
import os
import platform
import shutil
from enum import Enum

import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger("ConfigLoader")


class TorchDevice(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"

    @classmethod
    def values(cls):
        yield from (c.value for c in cls)


def default_device() -> TorchDevice:
    logger.info("No device specified in config - determining best device automatically")
    device = TorchDevice.CPU
    if torch.cuda.is_available():
        device = TorchDevice.CUDA
    elif torch.mps.is_available():
        device = TorchDevice.MPS

    logger.info(f"Using device: {device}")
    return device


def os_dependent_dataloader_kwargs() -> dict:
    kwargs = {}
    if platform.system() == "Darwin":
        # Considerable performance improvement avoiding spawn for dataloaders and persisting loaders on MacOSX
        kwargs = {"multiprocessing_context": "forkserver", "persistent_workers": True}

    return kwargs


def override_config(args, config):
    """Overrides config params with the ones given in command line.

    Args:
        args: Command line arguments.
        config: Configuration dictionary to override.
    """

    args_dict = vars(args)
    # remove the first argument which is the config file path
    args_dict.pop("config")

    for key, value in args_dict.items():
        if value is None:
            continue
        c = config
        for k in key.split("."):
            if k not in c:
                raise ValueError(f"Invalid config key: {key}")
            if isinstance(c[k], dict):
                c = c[k]
            else:
                c[k] = value


def load_config() -> tuple[dict, str]:
    parser = argparse.ArgumentParser(description="UNet3D")
    parser.add_argument("--config", type=str, help="Path to the YAML config file", required=True)
    # add additional command line arguments for the prediction that override the ones in the config file
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--loaders.output_dir", type=str, required=False)
    parser.add_argument("--loaders.test.file_paths", type=str, nargs="+", required=False)
    parser.add_argument("--loaders.test.slice_builder.patch_shape", type=int, nargs="+", required=False)
    parser.add_argument("--loaders.test.slice_builder.stride_shape", type=int, nargs="+", required=False)

    args = parser.parse_args()
    config_path = args.config
    config = _load_config_yaml(config_path)
    override_config(args, config)

    config_device = config.get("device", None)

    try:
        config["device"] = TorchDevice(config_device) if config_device is not None else default_device()
    except ValueError as e:
        raise ValueError(
            f"Config key device: {config_device} not understood -- supported values: {', '.join(TorchDevice.values())}"
        ) from e

    if config["device"] == TorchDevice.CPU:
        logger.warning("CPU mode will likely result in slow training/prediction")

    return config, config_path


def copy_config(config: dict, config_path: str):
    """Copies the config file to the checkpoint folder."""

    def _get_last_subfolder_path(path):
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        return max(subfolders, default=None)

    checkpoint_dir = os.path.join(config["trainer"].pop("checkpoint_dir"), "logs")
    last_run_dir = _get_last_subfolder_path(checkpoint_dir)
    config_file_name = os.path.basename(config_path)

    if last_run_dir:
        shutil.copy2(config_path, os.path.join(last_run_dir, config_file_name))


def _load_config_yaml(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)
