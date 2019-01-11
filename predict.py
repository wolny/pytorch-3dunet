import argparse
import os

import h5py
import numpy as np
import torch

from datasets.hdf5 import HDF5Dataset
from unet3d import utils
from unet3d.losses import SUPPORTED_LOSSES
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

    return probability_maps / normalization_mask


def save_predictions(probability_maps, output_file, average_channels):
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


def _final_sigmoid(loss):
    assert loss in SUPPORTED_LOSSES
    return loss == 'bce'


def main():
    parser = argparse.ArgumentParser(description='3D U-Net predictions')
    parser.add_argument('--model-path', required=True, type=str,
                        help='path to the model')
    parser.add_argument('--in-channels', required=True, type=int,
                        help='number of input channels')
    parser.add_argument('--out-channels', required=True, type=int,
                        help='number of output channels')
    parser.add_argument('--init-channel-number', type=int, default=64,
                        help='Initial number of feature maps in the encoder path which gets doubled on every stage (default: 64)')
    parser.add_argument('--interpolate',
                        help='use F.interpolate instead of ConvTranspose3d',
                        action='store_true')
    parser.add_argument('--average-channels',
                        help='average the probability_maps across the the channel axis (use only if your channels refer to the same semantic class)',
                        action='store_true')
    parser.add_argument('--layer-order', type=str,
                        help="Conv layer ordering, e.g. 'crg' -> Conv3D+ReLU+GroupNorm",
                        default='crg')
    parser.add_argument('--loss', type=str, required=True,
                        help='Loss function used for training. Possible values: [ce, bce, wce, dice]. Has to be provided cause loss determines the final activation of the model.')
    parser.add_argument('--test-path', type=str, required=True, help='path to the test dataset')
    parser.add_argument('--raw-internal-path', type=str, default='raw')
    parser.add_argument('--patch', required=True, type=int, nargs='+', default=None,
                        help='Patch shape for used for prediction on the test set')
    parser.add_argument('--stride', required=True, type=int, nargs='+', default=None,
                        help='Patch stride for used for prediction on the test set')

    args = parser.parse_args()

    # make sure those values correspond to the ones used during training
    in_channels = args.in_channels
    out_channels = args.out_channels
    # use F.interpolate for upsampling
    interpolate = args.interpolate
    layer_order = args.layer_order
    final_sigmoid = _final_sigmoid(args.loss)
    model = UNet3D(in_channels, out_channels,
                   init_channel_number=args.init_channel_number,
                   final_sigmoid=final_sigmoid,
                   interpolate=interpolate,
                   conv_layer_order=layer_order)

    logger.info(f'Loading model from {args.model_path}...')
    utils.load_checkpoint(args.model_path, model)

    logger.info('Loading datasets...')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        logger.warning('No CUDA device available. Using CPU for predictions')
        device = torch.device('cpu')

    model = model.to(device)

    patch = tuple(args.patch)
    stride = tuple(args.stride)

    dataset = HDF5Dataset(args.test_path, patch, stride, phase='test', raw_internal_path=args.raw_internal_path)
    probability_maps = predict(model, dataset, out_channels, device)

    output_file = f'{os.path.splitext(args.test_path)[0]}_probabilities.h5'

    save_predictions(probability_maps, output_file, args.average_channels)


if __name__ == '__main__':
    main()
