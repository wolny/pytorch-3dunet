import os

import h5py
import numpy as np
import torch

from datasets.hdf5 import get_test_datasets
from unet3d import utils
from unet3d.config import load_config
from unet3d.model import get_model

logger = utils.get_logger('UNet3DPredictor')


def predict(model, hdf5_dataset, config):
    """
    Return prediction masks by applying the model on the given dataset

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        hdf5_dataset (torch.utils.data.Dataset): input dataset
        out_channels (int): number of channels in the network output
        device (torch.Device): device to run the prediction on

    Returns:
         prediction_maps (numpy array): prediction masks for given dataset
    """

    def _volume_shape(hdf5_dataset):
        #TODO: support multiple internal datasets
        raw = hdf5_dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    out_channels = config['model']['out_channels']
    device = config['device']
    output_heads = config['model'].get('output_heads', 1)

    logger.info(f'Running prediction on {len(hdf5_dataset)} patches...')
    # dimensionality of the the output (CxDxHxW)
    volume_shape = _volume_shape(hdf5_dataset)
    prediction_maps_shape = (out_channels,) + volume_shape
    logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

    # initialize the output prediction arrays
    prediction_maps = [np.zeros(prediction_maps_shape, dtype='float32') for _ in range(output_heads)]
    # initialize normalization mask in order to average out probabilities of overlapping patches
    normalization_masks = [np.zeros(prediction_maps_shape, dtype='float32') for _ in range(output_heads)]

    # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
    model.eval()
    # Run predictions on the entire input dataset
    with torch.no_grad():
        for patch, index in hdf5_dataset:
            logger.info(f'Predicting slice:{index}')

            # save patch index: (C,D,H,W)
            channel_slice = slice(0, out_channels)
            index = (channel_slice,) + index

            # convert patch to torch tensor NxCxDxHxW and send to device we're using batch size of 1
            patch = patch.unsqueeze(dim=0).to(device)

            # forward pass
            predictions = model(patch)
            # wrap predictions into a list if there is only one output head from the network
            if output_heads == 1:
                predictions = [predictions]

            for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                      normalization_masks):
                # squeeze batch dimension and convert back to numpy array
                prediction = prediction.squeeze(dim=0).cpu().numpy()
                # unpad in order to avoid block artifacts in the output probability maps
                u_prediction, u_index = utils.unpad(prediction, index, volume_shape)
                # accumulate probabilities into the output prediction array
                prediction_map[u_index] += u_prediction
                # count voxel visits for normalization
                normalization_mask[u_index] += 1

    return [prediction_map / normalization_mask for prediction_map, normalization_mask in
            zip(prediction_maps, normalization_masks)]


def save_predictions(prediction_maps, output_file, dataset_names):
    """
    Saving probability maps to a given output H5 file. If 'average_channels'
    is set to True average the probability_maps across the the channel axis
    (useful in case where each channel predicts semantically the same thing).

    Args:
        prediction_maps (list): list of numpy array containing prediction maps in separate channels
        output_file (string): path to the output H5 file
        dataset_names (list): list of dataset names inside H5 file where the prediction maps will be saved
    """
    assert len(prediction_maps) == len(dataset_names), 'Each prediction map has to have a corresponding dataset name'
    logger.info(f'Saving predictions to: {output_file}...')

    with h5py.File(output_file, "w") as output_h5:
        for prediction_map, dataset_name in zip(prediction_maps, dataset_names):
            logger.info(f"Creating dataset '{dataset_name}'...")
            output_h5.create_dataset(dataset_name, data=prediction_map, compression="gzip")


def _get_output_file(dataset, suffix='_predictions'):
    return f'{os.path.splitext(dataset.file_path)[0]}{suffix}.h5'


def _get_dataset_names(config, number_of_datasets):
    dataset_names = config.get('dest_dataset_name')
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names
    else:
        default_prefix = 'predictions'
        if number_of_datasets == 1:
            return [default_prefix]
        else:
            return [f'{default_prefix}{i}' for i in range(number_of_datasets)]


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config)

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    model = model.to(config['device'])

    logger.info('Loading HDF5 datasets...')
    for test_dataset in get_test_datasets(config):
        logger.info(f"Processing '{test_dataset.file_path}'...")
        # run the model prediction on the entire dataset
        predictions = predict(model, test_dataset, config)
        # save the resulting probability maps
        output_file = _get_output_file(test_dataset)
        dataset_names = _get_dataset_names(config, len(predictions))
        save_predictions(predictions, output_file, dataset_names)


if __name__ == '__main__':
    main()
