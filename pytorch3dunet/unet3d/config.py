import argparse
import os
import shutil
import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger('ConfigLoader')


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

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
    return config, args.config


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
