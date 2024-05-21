import os
import time
from concurrent import futures
from pathlib import Path

import h5py
import numpy as np
import torch
from skimage import measure
from torch import nn
from tqdm import tqdm

from pytorch3dunet.datasets.hdf5 import AbstractHDF5Dataset
from pytorch3dunet.datasets.utils import SliceBuilder, remove_padding
from pytorch3dunet.unet3d.model import UNet2D
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('UNetPredictor')


def _get_output_file(dataset, suffix='_predictions', output_dir=None):
    input_dir, file_name = os.path.split(dataset.file_path)
    if output_dir is None:
        output_dir = input_dir
    output_filename = os.path.splitext(file_name)[0] + suffix + '.h5'
    return Path(output_dir) / output_filename


def _is_2d_model(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    return isinstance(model, UNet2D)


class _AbstractPredictor:
    def __init__(self,
                 model: nn.Module,
                 output_dir: str,
                 out_channels: int,
                 output_dataset: str = 'predictions',
                 save_segmentation: bool = False,
                 prediction_channel: int = None,
                 **kwargs):
        """
        Base class for predictors.
        Args:
            model: segmentation model
            output_dir: directory where the predictions will be saved
            out_channels: number of output channels of the model
            output_dataset: name of the dataset in the H5 file where the predictions will be saved
            save_segmentation: if true the segmentation will be saved instead of the probability maps
            prediction_channel: save only the specified channel from the network output
        """
        self.model = model
        self.output_dir = output_dir
        self.out_channels = out_channels
        self.output_dataset = output_dataset
        self.save_segmentation = save_segmentation
        self.prediction_channel = prediction_channel

    def __call__(self, test_loader):
        raise NotImplementedError


class StandardPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result as H5 file.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `output_dataset` config argument.
    """

    def __init__(self,
                 model: nn.Module,
                 output_dir: str,
                 out_channels: int,
                 output_dataset: str = 'predictions',
                 save_segmentation: bool = False,
                 prediction_channel: int = None,
                 **kwargs):
        super().__init__(model, output_dir, out_channels, output_dataset, save_segmentation, prediction_channel,
                         **kwargs)

    def __call__(self, test_loader):
        assert isinstance(test_loader.dataset, AbstractHDF5Dataset)
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")
        start = time.perf_counter()

        logger.info(f'Running inference on {len(test_loader)} batches')
        # dimensionality of the output predictions
        volume_shape = test_loader.dataset.volume_shape()
        if self.prediction_channel is not None:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape
        else:
            prediction_maps_shape = (self.out_channels,) + volume_shape

        # create destination H5 file
        output_file = _get_output_file(dataset=test_loader.dataset, output_dir=self.output_dir)
        with h5py.File(output_file, 'w') as h5_output_file:
            # allocate prediction and normalization arrays
            logger.info('Allocating prediction and normalization arrays...')
            prediction_map, normalization_mask = self._allocate_prediction_maps(prediction_maps_shape, h5_output_file)

            # determine halo used for padding
            patch_halo = test_loader.dataset.halo_shape

            # Sets the module in evaluation mode explicitly
            # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
            self.model.eval()
            # Run predictions on the entire input dataset
            with torch.no_grad():
                for input, indices in tqdm(test_loader):
                    # send batch to gpu
                    if torch.cuda.is_available():
                        input = input.pin_memory().cuda(non_blocking=True)

                    if _is_2d_model(self.model):
                        # remove the singleton z-dimension from the input
                        input = torch.squeeze(input, dim=-3)
                        # forward pass
                        prediction = self.model(input)
                        # add the singleton z-dimension to the output
                        prediction = torch.unsqueeze(prediction, dim=-3)
                    else:
                        # forward pass
                        prediction = self.model(input)

                    # unpad the predicted patch
                    prediction = remove_padding(prediction, patch_halo)
                    # convert to numpy array
                    prediction = prediction.cpu().numpy()
                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if self.prediction_channel is None:
                            channel_slice = slice(0, self.out_channels)
                        else:
                            # use only the specified channel
                            channel_slice = slice(0, 1)
                            pred = np.expand_dims(pred[self.prediction_channel], axis=0)

                        # add channel dimension to the index
                        index = (channel_slice,) + tuple(index)
                        # accumulate probabilities into the output prediction array
                        prediction_map[index] += pred
                        # count voxel visits for normalization
                        normalization_mask[index] += 1

            logger.info(f'Finished inference in {time.perf_counter() - start:.2f} seconds')
            # save results
            output_type = 'segmentation' if self.save_segmentation else 'probability maps'
            logger.info(f'Saving {output_type} to: {output_file}')
            self._save_results(prediction_map, normalization_mask, h5_output_file, test_loader.dataset)

    def _allocate_prediction_maps(self, output_shape, output_file):
        # initialize the output prediction arrays
        prediction_map = np.zeros(output_shape, dtype='float32')
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_mask = np.zeros(output_shape, dtype='uint8')
        return prediction_map, normalization_mask

    def _save_results(self, prediction_map, normalization_mask, output_file, dataset):
        result = prediction_map / normalization_mask
        if self.save_segmentation:
            result = np.argmax(result, axis=0).astype('uint16')
        output_file.create_dataset(self.output_dataset, data=result, compression="gzip")


class LazyPredictor(StandardPredictor):
    """
        Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
        Predicted patches are directly saved into the H5 and they won't be stored in memory. Since this predictor
        is slower than the `StandardPredictor` it should only be used when the predicted volume does not fit into RAM.
        """

    def __init__(self,
                 model: nn.Module,
                 output_dir: str,
                 out_channels: int,
                 output_dataset: str = 'predictions',
                 save_segmentation: bool = False,
                 prediction_channel: int = None,
                 **kwargs):
        super().__init__(model, output_dir, out_channels, output_dataset, save_segmentation, prediction_channel,
                         **kwargs)

    def _allocate_prediction_maps(self, output_shape, output_file):
        # allocate datasets for probability maps
        prediction_map = output_file.create_dataset(self.output_dataset,
                                                    shape=output_shape,
                                                    dtype='float32',
                                                    chunks=True,
                                                    compression='gzip')
        # allocate datasets for normalization masks
        normalization_mask = output_file.create_dataset('normalization',
                                                        shape=output_shape,
                                                        dtype='uint8',
                                                        chunks=True,
                                                        compression='gzip')
        return prediction_map, normalization_mask

    def _save_results(self, prediction_map, normalization_mask, output_file, dataset):
        z, y, x = prediction_map.shape[1:]
        # take slices which are 1/27 of the original volume
        patch_shape = (z // 3, y // 3, x // 3)
        if self.save_segmentation:
            output_file.create_dataset('segmentation', shape=(z, y, x), dtype='uint16', chunks=True, compression='gzip')

        for index in SliceBuilder._build_slices(prediction_map, patch_shape=patch_shape, stride_shape=patch_shape):
            logger.info(f'Normalizing slice: {index}')
            prediction_map[index] /= normalization_mask[index]
            # make sure to reset the slice that has been visited already in order to avoid 'double' normalization
            # when the patches overlap with each other
            normalization_mask[index] = 1
            # save segmentation
            if self.save_segmentation:
                output_file['segmentation'][index[1:]] = np.argmax(prediction_map[index], axis=0).astype('uint16')

        del output_file['normalization']
        if self.save_segmentation:
            del output_file[self.output_dataset]


class DSB2018Predictor(_AbstractPredictor):
    def __init__(self, model, output_dir, config, save_segmentation=True, pmaps_thershold=0.5, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)
        self.pmaps_threshold = pmaps_thershold
        self.save_segmentation = save_segmentation

    def _slice_from_pad(self, pad):
        if pad == 0:
            return slice(None, None)
        else:
            return slice(pad, -pad)

    def __call__(self, test_loader):
        # Sets the module in evaluation mode explicitly
        self.model.eval()
        # initial process pool for saving results to disk
        executor = futures.ProcessPoolExecutor(max_workers=32)
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for img, path in test_loader:
                # send batch to gpu
                if torch.cuda.is_available():
                    img = img.cuda(non_blocking=True)
                # forward pass
                pred = self.model(img)

                executor.submit(
                    dsb_save_batch,
                    self.output_dir,
                    path
                )

        print('Waiting for all predictions to be saved to disk...')
        executor.shutdown(wait=True)


def dsb_save_batch(output_dir, path, pred, save_segmentation=True, pmaps_thershold=0.5):
    def _pmaps_to_seg(pred):
        mask = (pred > pmaps_thershold)
        return measure.label(mask).astype('uint16')

    # convert to numpy array
    for single_pred, single_path in zip(pred, path):
        logger.info(f'Processing {single_path}')
        single_pred = single_pred.squeeze()

        # save to h5 file
        out_file = os.path.splitext(single_path)[0] + '_predictions.h5'
        if output_dir is not None:
            out_file = os.path.join(output_dir, os.path.split(out_file)[1])

        with h5py.File(out_file, 'w') as f:
            # logger.info(f'Saving output to {out_file}')
            f.create_dataset('predictions', data=single_pred, compression='gzip')
            if save_segmentation:
                f.create_dataset('segmentation', data=_pmaps_to_seg(single_pred), compression='gzip')
