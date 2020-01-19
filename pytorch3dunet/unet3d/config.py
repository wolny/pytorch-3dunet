import argparse

import torch
import yaml


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D training')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    # Get a device to train on
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    return config


def _load_config_yaml(config_file):
    return yaml.load(open(config_file, 'r'))
