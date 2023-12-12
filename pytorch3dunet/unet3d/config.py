import argparse
import os
import shutil

import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger('ConfigLoader')


def _override_config(args, config):
    """Overrides config params with the ones given in command line."""

    args_dict = vars(args)
    # remove the first argument which is the config file path
    args_dict.pop('config')

    for key, value in args_dict.items():
        if value is None:
            continue
        c = config
        for k in key.split('.'):
            if k not in c:
                raise ValueError(f'Invalid config key: {key}')
            if isinstance(c[k], dict):
                c = c[k]
            else:
                c[k] = value


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    # add additional command line arguments for the prediction that override the ones in the config file
    parser.add_argument('--model_path', type=str, required=False)
    parser.add_argument('--loaders.output_dir', type=str, required=False)
    parser.add_argument('--loaders.test.file_paths', type=str, nargs="+", required=False)
    parser.add_argument('--loaders.test.slice_builder.patch_shape', type=int, nargs="+", required=False)
    parser.add_argument('--loaders.test.slice_builder.stride_shape', type=int, nargs="+", required=False)

    args = parser.parse_args()
    config_path = args.config
    config = yaml.safe_load(open(config_path, 'r'))
    _override_config(args, config)

    device = config.get('device', None)
    if device == 'cpu':
        logger.warning('CPU mode forced in config, this will likely result in slow training/prediction')
        config['device'] = 'cpu'
        return config

    if torch.cuda.is_available():
        config['device'] = 'cuda'
    else:
        logger.warning('CUDA not available, using CPU')
        config['device'] = 'cpu'
    return config, config_path


def copy_config(config, config_path):
    """Copies the config file to the checkpoint folder."""

    def _get_last_subfolder_path(path):
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        return max(subfolders, default=None)

    checkpoint_dir = os.path.join(
        config['trainer'].pop('checkpoint_dir'), 'logs')
    last_run_dir = _get_last_subfolder_path(checkpoint_dir)
    config_file_name = os.path.basename(config_path)

    if last_run_dir:
        shutil.copy2(config_path, os.path.join(last_run_dir, config_file_name))


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
