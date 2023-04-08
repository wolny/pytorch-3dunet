import os
import time
from concurrent import futures

import h5py
import numpy as np
import torch
from skimage import measure
from torch import nn
from tqdm import tqdm

from pytorch3dunet.datasets.hdf5 import AbstractHDF5Dataset
from pytorch3dunet.datasets.utils import SliceBuilder
from pytorch3dunet.unet3d.model import UNet2D
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('UNetPredictor')


def _get_output_file(dataset, suffix='_predictions', output_dir=None):
    input_dir, file_name = os.path.split(dataset.file_path)
    if output_dir is None:
        output_dir = input_dir
    output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + suffix + '.h5')
    return output_file


def _get_dataset_name(config, prefix='predictions'):
    return config.get('dest_dataset_name', 'predictions')


def _is_2d_model(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    return isinstance(model, UNet2D)


class _AbstractPredictor:
    def __init__(self, model, output_dir, config, **kwargs):
        self.model = model
        self.output_dir = output_dir
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def volume_shape(dataset):
        raw = dataset.raw
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    def __call__(self, test_loader):
        raise NotImplementedError


class StandardPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result as H5 file.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `dest_dataset_name` config argument. If the argument is
    not present in the config 'predictions' is used as a default dataset name.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        output_dir (str): path to the output directory (optional)
        config (dict): global config dict
    """

    def __init__(self, model, output_dir, config, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)

    def __call__(self, test_loader):
        assert isinstance(test_loader.dataset, AbstractHDF5Dataset)
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")
        start = time.time()

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Saving only channel '{prediction_channel}' from the network output")

        logger.info(f'Running inference on {len(test_loader)} batches')

        # dimensionality of the output predictions
        volume_shape = self.volume_shape(test_loader.dataset)
        out_channels = self.config['model'].get('out_channels')
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        # evey patch will be mirror-padded with the following halo
        patch_halo = self.predictor_config.get('patch_halo', (4, 4, 4))
        if _is_2d_model(self.model):
            patch_halo = list(patch_halo)
            patch_halo[0] = 0

        # create destination H5 file
        output_file = _get_output_file(dataset=test_loader.dataset, output_dir=self.output_dir)
        h5_output_file = h5py.File(output_file, 'w')
        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        prediction_map, normalization_mask = self._allocate_prediction_maps(prediction_maps_shape, h5_output_file)

        # Sets the module in evaluation mode explicitly
        # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
        self.model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for input, indices in tqdm(test_loader):
                # send batch to gpu
                if torch.cuda.is_available():
                    input = input.cuda(non_blocking=True)

                input = _pad(input, patch_halo)

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

                # unpad
                prediction = _unpad(prediction, patch_halo)
                # convert to numpy array
                prediction = prediction.cpu().numpy()
                # for each batch sample
                for pred, index in zip(prediction, indices):
                    # save patch index: (C,D,H,W)
                    if prediction_channel is None:
                        channel_slice = slice(0, out_channels)
                    else:
                        # use only the specified channel
                        channel_slice = slice(0, 1)
                        pred = np.expand_dims(pred[prediction_channel], axis=0)

                    # add channel dimension to the index
                    index = (channel_slice,) + tuple(index)
                    # accumulate probabilities into the output prediction array
                    prediction_map[index] += pred
                    # count voxel visits for normalization
                    normalization_mask[index] += 1

        logger.info(f'Finished inference in {time.time() - start:.2f} seconds')
        # save results
        logger.info(f'Saving predictions to: {output_file}')
        self._save_results(prediction_map, normalization_mask, h5_output_file, test_loader.dataset)
        # close the output H5 file
        h5_output_file.close()

    def _allocate_prediction_maps(self, output_shape, output_file):
        # initialize the output prediction arrays
        prediction_map = np.zeros(output_shape, dtype='float32')
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_mask = np.zeros(output_shape, dtype='uint8')
        return prediction_map, normalization_mask

    def _save_results(self, prediction_map, normalization_mask, output_file, dataset):
        dataset_name = _get_dataset_name(self.config)
        prediction_map = prediction_map / normalization_mask
        output_file.create_dataset(dataset_name, data=prediction_map, compression="gzip")


def _pad(m, patch_halo):
    if patch_halo is not None:
        z, y, x = patch_halo
        return nn.functional.pad(m, (x, x, y, y, z, z), mode='reflect')
    return m


def _unpad(m, patch_halo):
    if patch_halo is not None:
        z, y, x = patch_halo
        if z == 0:
            return m[..., y:-y, x:-x]
        else:
            return m[..., z:-z, y:-y, x:-x]
    return m


class LazyPredictor(StandardPredictor):
    """
        Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
        Predicted patches are directly saved into the H5 and they won't be stored in memory. Since this predictor
        is slower than the `StandardPredictor` it should only be used when the predicted volume does not fit into RAM.

        The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
        not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
        of the output head from the network.

        Args:
            model (Unet3D): trained 3D UNet model used for prediction
            output_dir (str): path to the output directory (optional)
            config (dict): global config dict
        """

    def __init__(self, model, output_dir, config, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)

    def _allocate_prediction_maps(self, output_shape, output_file):
        dataset_name = _get_dataset_name(self.config)
        # allocate datasets for probability maps
        prediction_map = output_file.create_dataset(dataset_name, shape=output_shape, dtype='float32', chunks=True,
                                                    compression='gzip')
        # allocate datasets for normalization masks
        normalization_mask = output_file.create_dataset('normalization', shape=output_shape, dtype='uint8', chunks=True,
                                                        compression='gzip')
        return prediction_map, normalization_mask

    def _save_results(self, prediction_map, normalization_mask, output_file, dataset):
        z, y, x = prediction_map.shape[1:]
        # take slices which are 1/27 of the original volume
        patch_shape = (z // 3, y // 3, x // 3)
        for index in SliceBuilder._build_slices(prediction_map, patch_shape=patch_shape, stride_shape=patch_shape):
            logger.info(f'Normalizing slice: {index}')
            prediction_map[index] /= normalization_mask[index]
            # make sure to reset the slice that has been visited already in order to avoid 'double' normalization
            # when the patches overlap with each other
            normalization_mask[index] = 1
        del output_file['normalization']


class DSB2018Predictor(_AbstractPredictor):
    def __init__(self, model, output_dir, config, save_segmentation=True, pmaps_thershold=0.5, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)
        self.pmaps_thershold = pmaps_thershold
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
