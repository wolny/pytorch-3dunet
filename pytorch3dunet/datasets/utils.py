import collections

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from pytorch3dunet.unet3d.utils import get_logger, get_class

logger = get_logger('Dataset')


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the patch and stride shape.

    Args:
        raw_dataset (ndarray): raw data
        label_dataset (ndarray): ground truth labels
        weight_dataset (ndarray): weights for the labels
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
        kwargs: additional metadata
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs):
        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            # take the first element in the label_dataset to build slices
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset, patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, ignore_index=None,
                 threshold=0.6, slack_acceptance=0.01, **kwargs):
        super().__init__(raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs)
        if label_dataset is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = label_dataset[label_idx]
            if ignore_index is not None:
                patch = np.copy(patch)
                patch[patch == ignore_index] = 0
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return non_ignore_counts > threshold or rand_state.rand() < slack_acceptance

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        logger.info(f'Filtering slices...')
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


def _loader_classes(class_name):
    modules = [
        'pytorch3dunet.datasets.hdf5',
        'pytorch3dunet.datasets.dsb',
        'pytorch3dunet.datasets.utils'
    ]
    return get_class(class_name, modules)


def get_slice_builder(raws, labels, weight_maps, config):
    assert 'name' in config
    logger.info(f"Slice builder config: {config}")
    slice_builder_cls = _loader_classes(config['name'])
    return slice_builder_cls(raws, labels, weight_maps, **config)


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"

    train_datasets = dataset_class.create_datasets(loaders_config, phase='train')

    val_datasets = dataset_class.create_datasets(loaders_config, phase='val')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, pin_memory=True,
                          num_workers=num_workers)
    }


def get_test_loaders(config):
    """
    Returns test DataLoader.

    :return: generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    test_datasets = dataset_class.create_datasets(loaders_config, phase='test')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.file_path}...')
        if hasattr(test_dataset, 'prediction_collate'):
            collate_fn = test_dataset.prediction_collate
        else:
            collate_fn = default_prediction_collate

        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         collate_fn=collate_fn)


def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def calculate_stats(images, global_normalization=True):
    """
    Calculates min, max, mean, std given a list of nd-arrays
    """
    if global_normalization:
        # flatten first since the images might not be the same size
        flat = np.concatenate(
            [img.ravel() for img in images]
        )
        pmin, pmax, mean, std = np.percentile(flat, 1), np.percentile(flat, 99.6), np.mean(flat), np.std(flat)
    else:
        pmin, pmax, mean, std = None, None, None, None

    return {
        'pmin': pmin,
        'pmax': pmax,
        'mean': mean,
        'std': std
    }
