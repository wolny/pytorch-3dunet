import os

import h5py
import numpy as np
import torch

from datasets.hdf5 import get_test_loaders
from unet3d import utils
from unet3d.config import load_config
from unet3d.model import get_model

logger = utils.get_logger('UNet3DPredictor')


def predict_in_memory(model, data_loader, output_file, config):
    """
    Return prediction masks by applying the model on the given dataset

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict

    Returns:
         prediction_maps (numpy array): prediction masks for given dataset
    """

    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    out_channels = config['model'].get('out_channels')
    if out_channels is None:
        out_channels = config['model']['dt_out_channels']

    prediction_channel = config.get('prediction_channel', None)
    if prediction_channel is not None:
        logger.info(f"Using only channel '{prediction_channel}' from the network output")

    device = config['device']
    output_heads = config['model'].get('output_heads', 1)

    logger.info(f'Running prediction on {len(data_loader)} patches...')
    # dimensionality of the the output (CxDxHxW)
    volume_shape = _volume_shape(data_loader.dataset)
    if prediction_channel is None:
        prediction_maps_shape = (out_channels,) + volume_shape
    else:
        # single channel prediction map
        prediction_maps_shape = (1,) + volume_shape

    logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

    # initialize the output prediction arrays
    prediction_maps = [np.zeros(prediction_maps_shape, dtype='float32') for _ in range(output_heads)]
    # initialize normalization mask in order to average out probabilities of overlapping patches
    normalization_masks = [np.zeros(prediction_maps_shape, dtype='float32') for _ in range(output_heads)]

    # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
    model.eval()
    # Run predictions on the entire input dataset
    with torch.no_grad():
        for patch, index in data_loader:
            logger.info(f'Predicting slice:{index}')

            # save patch index: (C,D,H,W)
            if prediction_channel is None:
                channel_slice = slice(0, out_channels)
            else:
                channel_slice = slice(0, 1)

            index = (channel_slice,) + tuple(index)

            # send patch to device
            patch = patch.to(device)
            # forward pass
            predictions = model(patch)

            # wrap predictions into a list if there is only one output head from the network
            if output_heads == 1:
                predictions = [predictions]

            for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                      normalization_masks):
                # squeeze batch dimension and convert back to numpy array
                prediction = prediction.squeeze(dim=0).cpu().numpy()
                if prediction_channel is not None:
                    # use only the 'prediction_channel'
                    logger.info(f"Using channel '{prediction_channel}'...")
                    prediction = np.expand_dims(prediction[prediction_channel], axis=0)

                # unpad in order to avoid block artifacts in the output probability maps
                u_prediction, u_index = utils.unpad(prediction, index, volume_shape)
                # accumulate probabilities into the output prediction array
                prediction_map[u_index] += u_prediction
                # count voxel visits for normalization
                normalization_mask[u_index] += 1

    # save probability maps
    prediction_datasets = _get_dataset_names(config, output_heads, prefix='predictions')
    with h5py.File(output_file, 'w') as f:
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask
            logger.info(f'Saving predictions to: {output_file}/{prediction_dataset}...')
            f.create_dataset(prediction_dataset, data=prediction_map, compression="gzip")


def predict(model, data_loader, output_file, config):
    """
    Return prediction masks by applying the model on the given dataset.
    The predictions are saved in the output H5 file on a patch by patch basis.
    If your dataset fits into memory use predict_in_memory() which is much faster.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict

    """

    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    out_channels = config['model'].get('out_channels')
    if out_channels is None:
        out_channels = config['model']['dt_out_channels']

    prediction_channel = config.get('prediction_channel', None)
    if prediction_channel is not None:
        logger.info(f"Using only channel '{prediction_channel}' from the network output")

    device = config['device']
    output_heads = config['model'].get('output_heads', 1)

    logger.info(f'Running prediction on {len(data_loader)} patches...')

    # dimensionality of the the output (CxDxHxW)
    volume_shape = _volume_shape(data_loader.dataset)
    if prediction_channel is None:
        prediction_maps_shape = (out_channels,) + volume_shape
    else:
        # single channel prediction map
        prediction_maps_shape = (1,) + volume_shape

    logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

    with h5py.File(output_file, 'w') as f:
        # allocate datasets for probability maps
        prediction_datasets = _get_dataset_names(config, output_heads, prefix='predictions')
        prediction_maps = [
            f.create_dataset(dataset_name, shape=prediction_maps_shape, dtype='float32', chunks=True,
                             compression='gzip')
            for dataset_name in prediction_datasets]

        # allocate datasets for normalization masks
        normalization_datasets = _get_dataset_names(config, output_heads, prefix='normalization')
        normalization_masks = [
            f.create_dataset(dataset_name, shape=prediction_maps_shape, dtype='uint8', chunks=True,
                             compression='gzip')
            for dataset_name in normalization_datasets]

        # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
        model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for patch, index in data_loader:
                logger.info(f'Predicting slice:{index}')

                # save patch index: (C,D,H,W)
                if prediction_channel is None:
                    channel_slice = slice(0, out_channels)
                else:
                    channel_slice = slice(0, 1)

                index = (channel_slice,) + tuple(index)

                # send patch to device
                patch = patch.to(device)
                # forward pass
                predictions = model(patch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                          normalization_masks):
                    # squeeze batch dimension and convert back to numpy array
                    prediction = prediction.squeeze(dim=0).cpu().numpy()
                    if prediction_channel is not None:
                        # use only the 'prediction_channel'
                        logger.info(f"Using channel '{prediction_channel}'...")
                        prediction = np.expand_dims(prediction[prediction_channel], axis=0)

                    # unpad in order to avoid block artifacts in the output probability maps
                    u_prediction, u_index = utils.unpad(prediction, index, volume_shape)
                    # accumulate probabilities into the output prediction array
                    prediction_map[u_index] += u_prediction
                    # count voxel visits for normalization
                    normalization_mask[u_index] += 1

        # normalize the prediction_maps inside the H5
        for prediction_map, normalization_mask, prediction_dataset, normalization_dataset in zip(prediction_maps,
                                                                                                 normalization_masks,
                                                                                                 prediction_datasets,
                                                                                                 normalization_datasets):
            # TODO: iterate block by block
            # split the volume into 4 parts and load each into the memory separately
            logger.info(f'Normalizing {prediction_dataset}...')
            z, y, x = volume_shape
            mid_x = x // 2
            mid_y = y // 2
            prediction_map[:, :, 0:mid_y, 0:mid_x] /= normalization_mask[:, :, 0:mid_y, 0:mid_x]
            prediction_map[:, :, mid_y:, 0:mid_x] /= normalization_mask[:, :, mid_y:, 0:mid_x]
            prediction_map[:, :, 0:mid_y, mid_x:] /= normalization_mask[:, :, 0:mid_y, mid_x:]
            prediction_map[:, :, mid_y:, mid_x:] /= normalization_mask[:, :, mid_y:, mid_x:]
            logger.info(f'Deleting {normalization_dataset}...')
            del f[normalization_dataset]


def _get_output_file(dataset, suffix='_predictions'):
    return f'{os.path.splitext(dataset.file_path)[0]}{suffix}.h5'


def _get_dataset_names(config, number_of_datasets, prefix='predictions'):
    dataset_names = config.get('dest_dataset_name')
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names
    else:
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config)

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])

    logger.info('Loading HDF5 datasets...')
    store_predictions_in_memory = config.get('store_predictions_in_memory', True)
    if store_predictions_in_memory:
        logger.info('Predictions will be stored in memory. Make sure you have enough RAM for you dataset.')

    for test_loader in get_test_loaders(config):
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")

        output_file = _get_output_file(test_loader.dataset)
        # run the model prediction on the entire dataset and save to the 'output_file' H5
        if store_predictions_in_memory:
            predict_in_memory(model, test_loader, output_file, config)
        else:
            predict(model, test_loader, output_file, config)


if __name__ == '__main__':
    main()
