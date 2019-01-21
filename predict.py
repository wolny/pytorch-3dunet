import argparse
import os
import yaml
import glob

import h5py
import numpy as np
import torch

from datasets.hdf5 import HDF5Dataset
from unet3d import utils
from unet3d.model import UNet3D

logger = utils.get_logger('UNet3DPredictor')


def predict(model, dataset, out_channels, device):
    """
    Return prediction masks by applying the model on the given dataset

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        dataset (torch.utils.data.Dataset): input dataset
        out_channels (int): number of channels in the network output
        device (torch.Device): device to run the prediction on

    Returns:
         probability_maps (numpy array): prediction masks for given dataset
    """
    logger.info(f'Running prediction on {len(dataset)} patches...')
    # dimensionality of the the output (CxDxHxW)
    dataset_shape = dataset.raw.shape
    if len(dataset_shape) == 3:
        volume_shape = dataset_shape
    else:
        volume_shape = dataset_shape[1:]
    probability_maps_shape = (out_channels,) + volume_shape
    logger.info(f'Shape of the output probability map: {probability_maps_shape}')
    # initialize the output prediction array
    probability_maps = np.zeros(probability_maps_shape, dtype='float32')

    # initialize normalization mask in order to average out probabilities
    # of overlapping patches
    normalization_mask = np.zeros(probability_maps_shape, dtype='float32')

    # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
    model.eval()
    with torch.no_grad():
        for patch, index in dataset:
            logger.info(f'Predicting slice:{index}')

            # save patch index: (C,) + (D,H,W)
            channel_slice = slice(0, out_channels)
            index = (channel_slice,) + index

            # convert patch to torch tensor NxCxDxHxW and send to device
            # we're using batch size of 1
            patch = patch.view((1,) + patch.shape).to(device)

            # forward pass
            probs = model(patch)
            # convert back to numpy array
            probs = probs.squeeze().cpu().numpy()
            # for out_channel == 1 we need to expand back to 4D
            if probs.ndim == 3:
                probs = np.expand_dims(probs, axis=0)
            # unpad in order to avoid block artifacts in the output probability maps
            probs, index = utils.unpad(probs, index, volume_shape)
            # accumulate probabilities into the output prediction array
            probability_maps[index] += probs
            # count voxel visits for normalization
            normalization_mask[index] += 1

    return probability_maps / (normalization_mask + 1e-16)


def save_predictions(probability_maps, output_file, output_file_uint8, average_channels):
    """
    Saving probability maps to a given output H5 file. If 'average_channels'
    is set to True average the probability_maps across the the channel axis
    (useful in case where each channel predicts semantically the same thing).

    Args:
        probability_maps (numpy.ndarray): numpy array containing probability
            maps for each class in separate channels
        output_file (string): path to the output H5 file
        average_channels (bool): if True average out the channels in the probability_maps otherwise
            keep the channels separate
    """
    logger.info(f'Saving predictions to: {output_file}')

    with h5py.File(output_file, "w") as output_h5:
        if average_channels:
            probability_maps = np.mean(probability_maps, axis=0)
        dataset_name = 'probability_maps'
        logger.info(f"Creating dataset '{dataset_name}'")

        output_h5.create_dataset(dataset_name, data=probability_maps, dtype=probability_maps.dtype, compression="gzip")

    with h5py.File(output_file_uint8, "w") as output_h5:
        # cast to uint8
        dataset_name = 'prediction'
        logger.info(f"Creating dataset '{dataset_name}'")
        probability_image = (255 * probability_maps[1] / np.max(probability_maps[1])).astype(np.uint8)
        output_h5.create_dataset(dataset_name, data=probability_image, dtype=np.uint8, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description='3D U-Net predictions')
    parser.add_argument('--config', required=True, type=str, help='Config file path')
    parser.add_argument('--test-path', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--model-path', type=str, required=False, help='Path to saved model')
    parser.add_argument('--save-path', type=str, help='Path to saving directory')

    parser.add_argument('--patch', required=False, type=int, nargs='+', default=None,
                        help='Patch shape for used for prediction on the test set')
    parser.add_argument('--stride', required=False, type=int, nargs='+', default=None,
                        help='Patch stride for used for prediction on the test set')
    args = parser.parse_args()

    config = yaml.load(open(parser.parse_args().config))

    # make sure those values correspond to the ones used during training
    in_channels = config['in-channels']
    out_channels = config['out-channels']

    # use F.interpolate for upsampling
    interpolate = config['interpolate']
    layer_order = config['layer-order']
    final_sigmoid = config['final-sigmoid']
    model = UNet3D(in_channels, out_channels,
                   init_channel_number=config['init-channel-number'],
                   final_sigmoid=final_sigmoid,
                   interpolate=interpolate,
                   conv_layer_order=layer_order)

    if args.model_path is None:
        model_path = config['checkpoint-dir'] + "/last_checkpoint.pytorch"
    else:
        model_path = args.model_path

    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)

    logger.info('Loading datasets...')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        logger.warning('No CUDA device available. Using CPU for predictions')
        device = torch.device('cpu')

    model = model.to(device)

    if args.patch is None:
        patch = tuple(config['val-patch'])
    else:
        patch = tuple(args.patch)

    if args.patch is None:
        stride = tuple(config['val-stride'])
    else:
        stride = tuple(args.stride)

    if os.path.isdir(args.test_path):
        test_paths = glob.glob(args.test_path + '**/*.h5', recursive=True)
    elif os.path.isfile(args.test_path):
        test_paths = [args.test_path]
    else:
        test_paths = []

    for test_path in test_paths:
        logger.info(f"Processing dataset '{test_path}'")
        dataset = HDF5Dataset(test_path, patch, stride, phase='test', raw_internal_path=config['raw-internal-path'])
        probability_maps = predict(model, dataset, out_channels, device)

        if args.save_path is None:
            output_file = f'{os.path.splitext(test_path)[0]}_probabilities.h5'
        else:
            output_file = args.save_path + '_probabilities.h5'

        if args.save_path is None:
            output_file_uint8 = f'{os.path.splitext(test_path)[0]}_prediction.h5'
        else:
            output_file_uint8 = args.save_path + '_prediction.h5'

        # average channels only in case of final_sigmoid
        save_predictions(probability_maps, output_file, output_file_uint8, final_sigmoid)


if __name__ == '__main__':
    main()
