import h5py
import numpy as np
import torch

from datasets.hdf5 import SliceBuilder
from unet3d.utils import get_logger
from unet3d.utils import unpad

logger = get_logger('UNet3DTrainer')


class _AbstractPredictor:
    def __init__(self, model, loader, output_file, config, **kwargs):
        self.model = model
        self.loader = loader
        self.output_file = output_file
        self.config = config

    @staticmethod
    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def _get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def predict(self):
        raise NotImplementedError


class StandardPredictor(_AbstractPredictor):
    """
        Applies the model on the given dataset and saves the result in the `output_file` in the H5 format.
        Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
        use `LazyPredictor` instead.

        The output dataset names inside the H5 is given by `des_dataset_name` config argument. If the argument is
        not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
        of the output head from the network.

        Args:
            model (Unet3D): trained 3D UNet model used for prediction
            data_loader (torch.utils.data.DataLoader): input data loader
            output_file (str): path to the output H5 file
            config (dict): global config dict
        """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def predict(self):
        out_channels = self.config['model'].get('out_channels')
        if out_channels is None:
            out_channels = self.config['model']['dt_out_channels']

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Using only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        logger.info(f'Running prediction on {len(self.loader)} patches...')

        # dimensionality of the the output predictions
        volume_shape = self._volume_shape(self.loader.dataset)
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        # create destination H5 file
        h5_output_file = h5py.File(self.output_file, 'w')
        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
                                                                              output_heads, h5_output_file)

        # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
        self.model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for patch, index in self.loader:
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
                predictions = self.model(patch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                          normalization_masks):
                    # squeeze batch dimension and convert back to numpy array
                    assert prediction.size()[0] == 1, 'Only batch size of 1 supported during prediction'
                    prediction = prediction.squeeze(dim=0).cpu().numpy()
                    if prediction_channel is not None:
                        # use only the 'prediction_channel'
                        logger.info(f"Using channel '{prediction_channel}'...")
                        prediction = np.expand_dims(prediction[prediction_channel], axis=0)

                    # unpad in order to avoid block artifacts in the output probability maps
                    u_prediction, u_index = unpad(prediction, index, volume_shape)
                    # accumulate probabilities into the output prediction array
                    prediction_map[u_index] += u_prediction
                    # count voxel visits for normalization
                    normalization_mask[u_index] += 1

        # save results to
        self._save_results(prediction_maps, normalization_masks, output_heads, h5_output_file)
        # close the output H5 file
        h5_output_file.close()

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file):
        # save probability maps
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask
            logger.info(f'Saving predictions to: {output_file}/{prediction_dataset}...')
            output_file.create_dataset(prediction_dataset, data=prediction_map, compression="gzip")


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
            data_loader (torch.utils.data.DataLoader): input data loader
            output_file (str): path to the output H5 file
            config (dict): global config dict
        """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # allocate datasets for probability maps
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        prediction_maps = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='float32', chunks=True,
                                       compression='gzip')
            for dataset_name in prediction_datasets]

        # allocate datasets for normalization masks
        normalization_datasets = self._get_output_dataset_names(output_heads, prefix='normalization')
        normalization_masks = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='uint8', chunks=True,
                                       compression='gzip')
            for dataset_name in normalization_datasets]

        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file):
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        normalization_datasets = self._get_output_dataset_names(output_heads, prefix='normalization')

        # normalize the prediction_maps inside the H5
        for prediction_map, normalization_mask, prediction_dataset, normalization_dataset in zip(prediction_maps,
                                                                                                 normalization_masks,
                                                                                                 prediction_datasets,
                                                                                                 normalization_datasets):
            # split the volume into 4 parts and load each into the memory separately
            logger.info(f'Normalizing {prediction_dataset}...')

            z, y, x = prediction_map.shape[1:]
            # take slices which are 1/27 of the original volume
            patch_shape = (z // 3, y // 3, x // 3)
            for index in SliceBuilder._build_slices(prediction_map, patch_shape=patch_shape, stride_shape=patch_shape):
                logger.info(f'Normalizing slice: {index}')
                prediction_map[index] /= normalization_mask[index]
                # make sure to reset the slice that has been visited already in order to avoid 'double' normalization
                # when the patches overlap with each other
                normalization_mask[index] = 1

            logger.info(f'Deleting {normalization_dataset}...')
            del output_file[normalization_dataset]


class EmbeddingsPredictor(_AbstractPredictor):
    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def predict(self):
        pass
