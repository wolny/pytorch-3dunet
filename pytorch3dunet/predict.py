import importlib
import os
import numpy as np

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model

logger = utils.get_logger('UNet3DPredict')


def get_predictor(model, config):
    output_dir = config['loaders'].get('output_dir', None)
    # override output_dir if provided in the 'predictor' section of the config
    output_dir = config.get('predictor', {}).get('output_dir', output_dir)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)
    out_channels = config['model'].get('out_channels')
    return predictor_class(model, output_dir, out_channels, **predictor_config)


def main():
    # Load configuration
    config, _ = load_config()

    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available

    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
    if torch.cuda.is_available() and not config['device'] == 'cpu':
        model = model.cuda()

    # create predictor instance
    predictor = get_predictor(model, config)

    for test_loader in get_test_loaders(config):
        # run the model prediction on the test_loader and save the results in the output_dir
        predictor(test_loader)

    truth = np.array(predictor.ground_truth)
    pred = np.array(predictor.predicted_label)

    print('ground truth', np.argmax(truth, 1))
    print('predicted', np.argmax(pred, 1))

if __name__ == '__main__':
    main()
