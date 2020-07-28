import os
import time

import h5py
import hdbscan
import numpy as np
import torch
from PIL import Image
from skimage import measure
from sklearn.cluster import MeanShift

from pytorch3dunet.datasets.hdf5 import AbstractHDF5Dataset
from pytorch3dunet.datasets.sliced import SlicedDataset
from pytorch3dunet.datasets.utils import SliceBuilder
from pytorch3dunet.embeddings.utils import MeanEmbeddingAnchor, RandomEmbeddingAnchor
from pytorch3dunet.unet3d.losses import _AbstractContrastiveLoss
from pytorch3dunet.unet3d.utils import get_logger, pca_project
from pytorch3dunet.unet3d.utils import remove_halo

logger = get_logger('UNetPredictor')


def _get_output_file(dataset, suffix='_predictions', output_dir=None):
    input_dir, file_name = os.path.split(dataset.file_path)
    if output_dir is None:
        output_dir = input_dir
    output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + suffix + '.h5')
    return output_file


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


class _AbstractPredictor:
    def __init__(self, model, output_dir, config, **kwargs):
        self.model = model
        self.output_dir = output_dir
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def __call__(self, test_loader):
        raise NotImplementedError


class StandardPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result as H5 file.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `dest_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.

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
        output_file = _get_output_file(dataset=test_loader.dataset, output_dir=self.output_dir)

        out_channels = self.config['model'].get('out_channels')

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Saving only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        logger.info(f'Running prediction on {len(test_loader)} batches...')

        # dimensionality of the the output predictions
        volume_shape = self.volume_shape(test_loader.dataset)
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        patch_halo = self.predictor_config.get('patch_halo', (4, 8, 8))
        self._validate_halo(patch_halo, self.config['loaders']['test']['slice_builder'])
        logger.info(f'Using patch_halo: {patch_halo}')

        # create destination H5 file
        h5_output_file = h5py.File(output_file, 'w')
        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
                                                                              output_heads, h5_output_file)

        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in test_loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                predictions = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                # for each output head
                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                          normalization_masks):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        index = (channel_slice,) + index

                        if prediction_channel is not None:
                            # use only the 'prediction_channel'
                            logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)

                        logger.info(f'Saving predictions for slice:{index}...')

                        # remove halo in order to avoid block artifacts in the output probability maps
                        u_prediction, u_index = remove_halo(pred, index, volume_shape, patch_halo)
                        # accumulate probabilities into the output prediction array
                        prediction_map[u_index] += u_prediction
                        # count voxel visits for normalization
                        normalization_mask[u_index] += 1

        # save results
        logger.info(f'Saving predictions to: {output_file}')
        self._save_results(prediction_maps, normalization_masks, output_heads, h5_output_file, test_loader.dataset)
        # close the output H5 file
        h5_output_file.close()

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset):
        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)

        # save probability maps
        prediction_datasets = self.get_output_dataset_names(output_heads, prefix='predictions')
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask

            if dataset.mirror_padding is not None:
                z_s, y_s, x_s = [_slice_from_pad(p) for p in dataset.mirror_padding]

                logger.info(f'Dataset loaded with mirror padding: {dataset.mirror_padding}. Cropping before saving...')

                prediction_map = prediction_map[:, z_s, y_s, x_s]

            output_file.create_dataset(prediction_dataset, data=prediction_map, compression="gzip")

    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"


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

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # allocate datasets for probability maps
        prediction_datasets = self.get_output_dataset_names(output_heads, prefix='predictions')
        prediction_maps = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='float32', chunks=True,
                                       compression='gzip')
            for dataset_name in prediction_datasets]

        # allocate datasets for normalization masks
        normalization_datasets = self.get_output_dataset_names(output_heads, prefix='normalization')
        normalization_masks = [
            output_file.create_dataset(dataset_name, shape=output_shape, dtype='uint8', chunks=True,
                                       compression='gzip')
            for dataset_name in normalization_datasets]

        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset):
        if dataset.mirror_padding:
            logger.warn(
                f'Mirror padding unsupported in LazyPredictor. Output predictions will be padded with pad_width: {dataset.pad_width}')

        prediction_datasets = self.get_output_dataset_names(output_heads, prefix='predictions')
        normalization_datasets = self.get_output_dataset_names(output_heads, prefix='normalization')

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


class DSB2018Predictor(_AbstractPredictor):
    def __init__(self, model, output_dir, config, save_segmentation=False, pmaps_thershold=0.5, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)
        self.pmaps_thershold = pmaps_thershold
        self.save_segmentation = save_segmentation

    def _slice_from_pad(self, pad):
        if pad == 0:
            return slice(None, None)
        else:
            return slice(pad, -pad)

    def __call__(self, test_loader):
        device = self.config['device']
        # Sets the module in evaluation mode explicitly
        self.model.eval()
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for img, path in test_loader:
                # send batch to device
                img = img.to(device)
                # forward pass
                pred = self.model(img)
                # convert to numpy array
                for single_pred, single_path in zip(pred, path):
                    logger.info(f'Processing {single_path}')
                    single_pred = single_pred.cpu().numpy().squeeze()

                    if hasattr(test_loader.dataset, 'mirror_padding') \
                            and test_loader.dataset.mirror_padding is not None:
                        z_s, y_s, x_s = [self._slice_from_pad(p) for p in test_loader.dataset.mirror_padding]
                        single_pred = single_pred[y_s, x_s]

                    # save to h5 file
                    out_file = os.path.splitext(single_path)[0] + '_predictions.h5'
                    if self.output_dir is not None:
                        out_file = os.path.join(self.output_dir, os.path.split(out_file)[1])

                    with h5py.File(out_file, 'w') as f:
                        logger.info(f'Saving output to {out_file}')
                        f.create_dataset('predictions', data=single_pred, compression='gzip')
                        if self.save_segmentation:
                            f.create_dataset('segmentation', data=self._pmaps_to_seg(single_pred), compression='gzip')

    def _pmaps_to_seg(self, pred):
        mask = (pred > self.pmaps_thershold).astype('uint8')
        return measure.label(mask)


class AnchorEmbeddingsPredictor(_AbstractPredictor):
    def __init__(self, model, output_dir, config, epsilon, anchor_extraction='mean', norm='fro', **kwargs):
        super().__init__(model, output_dir, config, **kwargs)

        self.epsilon = epsilon
        self.anchor_extraction = anchor_extraction

        if anchor_extraction == 'mean':
            # function for computing a mean embeddings of target instances
            c_mean_fn = _AbstractContrastiveLoss._compute_cluster_means
            self.anchor_embeddings_extractor = MeanEmbeddingAnchor(c_mean_fn)
        else:
            self.anchor_embeddings_extractor = RandomEmbeddingAnchor()

        self.norm = norm

    def __call__(self, test_loader):
        assert isinstance(test_loader.dataset, SlicedDataset)
        device = self.config['device']
        # Sets the module in evaluation mode explicitly
        self.model.eval()
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for img, tar, path in test_loader:
                logger.info(f'Processing {path}')
                # send batch to device
                img = img.to(device)
                tar = tar.to(device)
                # forward pass
                emb = self.model(img)

                seg = self._emb_to_seg(emb, tar)

                for single_emb, single_seg, single_path in zip(emb, seg, path):
                    # save to h5 file
                    out_file = os.path.splitext(single_path)[0] + '_predictions.h5'
                    if self.output_dir is not None:
                        out_file = os.path.join(self.output_dir, os.path.split(out_file)[1])

                    # save PNG with PCA projected embeddings
                    embeddings_numpy = np.squeeze(single_emb.cpu().numpy())
                    rgb_img = pca_project(embeddings_numpy)
                    Image.fromarray(np.rollaxis(rgb_img, 0, 3)).save(os.path.splitext(out_file)[0] + '.png')

                    with h5py.File(out_file, 'w') as f:
                        logger.info(f'Saving output to {out_file}')
                        f.create_dataset('segmentation', data=np.uint16(single_seg.cpu().numpy()), compression='gzip')

    def _emb_to_seg(self, embeddings, target):
        # iterate over batch dimension
        results = []
        for emb, tar in zip(embeddings, target):
            anchor_embeddings = self.anchor_embeddings_extractor(emb, tar)
            result = torch.zeros_like(tar)
            for i, cm in enumerate(anchor_embeddings):
                if i == 0:
                    # ignore 0-label
                    continue

                # compute distance map; embeddings is ExSPATIAL, cluster_mean is ExSINGLETON_SPATIAL, so we can just broadcast
                inst_mask = torch.norm(emb - cm, self.norm, dim=0) < self.epsilon
                # save instance
                result[inst_mask] = i + 1

            results.append(result)

        return torch.stack(results, dim=0)


class EmbeddingsPredictor(_AbstractPredictor):
    """
    Applies the embedding model on the given dataset and saves the result in the `output_file` in the H5 format.

    The resulting volume is the segmentation itself (not the embedding vectors) obtained by clustering embeddings
    with HDBSCAN or MeanShift algorithm patch by patch and then stitching the patches together.
    """

    def __init__(self, model, output_dir, config, clustering, iou_threshold=0.7, noise_label=-1, **kwargs):
        super().__init__(model, output_dir, config, **kwargs)

        self.iou_threshold = iou_threshold
        self.noise_label = noise_label
        self.clustering = clustering

        assert clustering in ['hdbscan', 'meanshift'], 'Only HDBSCAN and MeanShift are supported'
        logger.info(f'IoU threshold: {iou_threshold}')

        self.clustering_name = clustering
        self.clustering = self._get_clustering(clustering, kwargs)

    def __call__(self, test_loader):
        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        logger.info(f'Running prediction on {len(test_loader)} patches...')

        # dimensionality of the the output segmentation
        volume_shape = self.volume_shape(test_loader.dataset)

        logger.info(f'The shape of the output segmentation (DHW): {volume_shape}')

        logger.info('Allocating segmentation array...')
        # initialize the output prediction arrays
        output_segmentations = [np.zeros(volume_shape, dtype='int32') for _ in range(output_heads)]
        # initialize visited_voxels arrays
        visited_voxels_arrays = [np.zeros(volume_shape, dtype='uint8') for _ in range(output_heads)]

        # Sets the module in evaluation mode explicitly
        self.model.eval()
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in test_loader:
                # logger.info(f'Predicting embeddings for slice:{index}')

                # send batch to device
                batch = batch.to(device)
                # forward pass
                embeddings = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    embeddings = [embeddings]

                for prediction, output_segmentation, visited_voxels_array in zip(embeddings, output_segmentations,
                                                                                 visited_voxels_arrays):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # iterate sequentially because of the current simple stitching that we're using
                    for pred, index in zip(prediction, indices):
                        # convert embeddings to segmentation with hdbscan clustering
                        segmentation = self._embeddings_to_segmentation(pred)
                        # stitch patches
                        self._merge_segmentation(segmentation, index, output_segmentation, visited_voxels_array)

        # save results
        output_file = _get_output_file(dataset=test_loader.dataset, output_dir=self.output_dir)
        with h5py.File(output_file, 'w') as f:
            prediction_datasets = self.get_output_dataset_names(output_heads,
                                                                prefix=f'segmentation/{self.clustering_name}')
            for output_segmentation, prediction_dataset in zip(output_segmentations, prediction_datasets):
                logger.info(f'Saving predictions to: {output_file}/{prediction_dataset}...')
                f.create_dataset(prediction_dataset, data=output_segmentation, compression="gzip")

    def _embeddings_to_segmentation(self, embeddings):
        """
        Cluster embeddings vectors with HDBSCAN and return the segmented volume.

        Args:
            embeddings (ndarray): 4D (CDHW) embeddings tensor
        Returns:
            3D (DHW) segmentation
        """
        # shape of the output segmentation
        output_shape = embeddings.shape[1:]
        # reshape (C, D, H, W) -> (C, D * H * W) and transpose -> (D * H * W, C)
        flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()

        logger.info('Clustering embeddings...')
        # perform clustering and reshape in order to get the segmentation volume
        start = time.time()
        clusters = self.clustering.fit_predict(flattened_embeddings).reshape(output_shape)
        logger.info(
            f'Number of clusters found by {self.clustering}: {np.max(clusters)}. Duration: {time.time() - start} sec.')
        return clusters

    def _merge_segmentation(self, segmentation, index, output_segmentation, visited_voxels_array):
        """
        Given the `segmentation` patch, its `index` in the `output_segmentation` array and the array visited voxels
        merge the segmented patch (`segmentation`) into the `output_segmentation`

        Args:
            segmentation (ndarray): segmented patch
            index (tuple): position of the patch inside `output_segmentation` volume
            output_segmentation (ndarray): current state of the output segmentation
            visited_voxels_array (ndarray): array of voxels visited so far (same size as `output_segmentation`); visited
                voxels will be marked by a number greater than 0
        """
        index = tuple(index)
        # get new unassigned label
        max_label = np.max(output_segmentation) + 1
        # make sure there are no clashes between current segmentation patch and the output_segmentation
        # but keep the noise label
        noise_mask = segmentation == self.noise_label
        segmentation += int(max_label)
        segmentation[noise_mask] = self.noise_label
        # get the overlap mask in the current patch
        overlap_mask = visited_voxels_array[index] > 0
        # get the new labels inside the overlap_mask
        new_labels = np.unique(segmentation[overlap_mask])
        merged_labels = self._merge_labels(output_segmentation[index], new_labels, segmentation)
        # relabel new segmentation with the merged labels
        for current_label, new_label in merged_labels:
            segmentation[segmentation == new_label] = current_label
        # update the output_segmentation
        output_segmentation[index] = segmentation
        # visit the patch
        visited_voxels_array[index] += 1

    def _merge_labels(self, current_segmentation, new_labels, new_segmentation):
        def _most_frequent_label(labels):
            unique, counts = np.unique(labels, return_counts=True)
            ind = np.argmax(counts)
            return unique[ind]

        result = []
        # iterate over new_labels and merge regions if the IoU exceeds a given threshold
        for new_label in new_labels:
            # skip 'noise' label assigned by hdbscan
            if new_label == self.noise_label:
                continue
            new_label_mask = new_segmentation == new_label
            # get only the most frequent overlapping label
            most_frequent_label = _most_frequent_label(current_segmentation[new_label_mask])
            # skip 'noise' label
            if most_frequent_label == self.noise_label:
                continue
            current_label_mask = current_segmentation == most_frequent_label
            # compute Jaccard index
            iou = np.bitwise_and(new_label_mask, current_label_mask).sum() / np.bitwise_or(new_label_mask,
                                                                                           current_label_mask).sum()
            if iou > self.iou_threshold:
                # merge labels
                result.append((most_frequent_label, new_label))

        return result

    def _get_clustering(self, clustering_alg, kwargs):
        logger.info(f'Using {clustering_alg} for clustering')

        if clustering_alg == 'hdbscan':
            min_cluster_size = kwargs.get('min_cluster_size', 50)
            min_samples = kwargs.get('min_samples', None),
            metric = kwargs.get('metric', 'euclidean')
            cluster_selection_method = kwargs.get('cluster_selection_method', 'eom')

            logger.info(f'HDBSCAN params: min_cluster_size: {min_cluster_size}, min_samples: {min_samples}')
            return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,
                                   cluster_selection_method=cluster_selection_method)
        else:
            bandwidth = kwargs['bandwidth']
            logger.info(f'MeanShift params: bandwidth: {bandwidth}, bin_seeding: True')
            # use fast MeanShift with bin seeding
            return MeanShift(bandwidth=bandwidth, bin_seeding=True)
