import collections
from typing import Any, Optional

import numpy as np
import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from pytorch3dunet.unet3d.utils import get_logger, get_class

logger = get_logger('Dataset')


class RandomScaler:
    """
    Randomly scales the raw and label patches.

    Args:
        scale_range (int): the maximum absolute value of the scaling factor,
            i.e. patches coordinates will be randomly shifted in the range [-scale_range, scale_range]
        patch_shape (tuple): the shape of the patch DxHxW
        volume_shape (tuple): the shape of the volume DxHxW
        execution_probability (float): the probability of executing the scaling
        seed (int): random seed
    """

    def __init__(
            self,
            scale_range: int,
            patch_shape: tuple,
            volume_shape: tuple,
            execution_probability: float = 0.5,
            seed: int = 47
    ):
        self.scale_range = scale_range
        self.patch_shape = patch_shape
        self.volume_shape = volume_shape
        self.execution_probability = execution_probability
        self.rs = np.random.RandomState(seed)

    def randomize_indices(self, raw_idx: tuple, label_idx: tuple) -> tuple[tuple, tuple]:
        # execute scaling with a given probability
        if self.rs.uniform() < self.execution_probability:
            return raw_idx, label_idx

        # select random offsets for scaling
        offsets = [self.rs.randint(self.scale_range) for _ in range(3)]
        # change offset sign at random
        if self.rs.rand() > 0.5:
            offsets = [-o for o in offsets]
        # apply offsets to the start or end of the slice at random
        is_start = self.rs.rand() > 0.5
        raw_idx = self._apply_offsets(raw_idx, offsets, is_start)
        label_idx = self._apply_offsets(label_idx, offsets, is_start)

        # assert spatial dimensions are the same
        if len(raw_idx) == 4:
            raw_idx_spacial = raw_idx[1:]
        else:
            raw_idx_spacial = raw_idx
        if len(label_idx) == 4:
            label_idx_spacial = label_idx[1:]
        else:
            label_idx_spacial = label_idx
        assert raw_idx_spacial == label_idx_spacial, f"Raw and label indices are different: {raw_idx_spacial} != {label_idx_spacial}"

        return raw_idx, label_idx

    def rescale_patches(self, raw_patch: torch.Tensor, label_patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # compute zoom factors
        if raw_patch.ndim == 4:
            raw_shape = raw_patch.shape[1:]
        else:
            raw_shape = raw_patch.shape

        # if raw_shape equal to self.patch_shape just return the patches
        if raw_shape == self.patch_shape:
            return raw_patch, label_patch

        # rescale patches back to the original shape
        if raw_patch.ndim == 4:
            # add batch dimension
            raw_patch = raw_patch.unsqueeze(0)
            remove_dims = 1
        else:
            # add batch and channels dimensions
            raw_patch = raw_patch.unsqueeze(0).unsqueeze(0)
            remove_dims = 2

        # interpolate raw patch
        raw_patch = interpolate(raw_patch, self.patch_shape, mode='trilinear')
        # remove additional dimensions
        for _ in range(remove_dims):
            raw_patch = raw_patch.squeeze(0)

        if label_patch.ndim == 4:
            label_patch = label_patch.unsqueeze(0)
            remove_dims = 1
        else:
            label_patch = label_patch.unsqueeze(0).unsqueeze(0)
            remove_dims = 2

        label_dtype = label_patch.dtype
        # check if label patch is of torch int type
        if label_dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            # convert to float for interpolation
            label_patch = label_patch.float()

        # interpolate label patch
        label_patch = interpolate(label_patch, self.patch_shape, mode='nearest')

        # remove additional dimensions
        for _ in range(remove_dims):
            label_patch = label_patch.squeeze(0)

        # convert back to int if necessary
        if label_dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            if label_dtype == torch.int64:
                label_patch = label_patch.long()
            else:
                label_patch = label_patch.int()

        return raw_patch, label_patch

    def _apply_offsets(self, idx: tuple, offsets: list, is_start: bool) -> tuple:
        if len(idx) == 4:
            spatial_idx = idx[1:]
        else:
            spatial_idx = idx

        new_idx = []
        for i, o, s in zip(spatial_idx, offsets, self.volume_shape):
            if is_start:
                # prevent negative start
                start = max(0, i.start + o)
                stop = i.stop
            else:
                start = i.start
                # prevent stop exceeding the volume shape
                stop = min(s, i.stop + o)

            new_idx.append(slice(start, stop))

        if len(idx) == 4:
            return (idx[0],) + tuple(new_idx)

        return tuple(new_idx)


class ConfigDataset(Dataset):
    """
    Abstract class for datasets that are configured via a dictionary.
    """

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
    Builds the position of the patches in a given raw/label ndarray based on the patch and stride shape.

    Args:
        raw_dataset (ndarray): raw data
        label_dataset (ndarray): ground truth labels
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
        kwargs: additional metadata
    """

    def __init__(
            self,
            raw_dataset: np.ndarray,
            label_dataset: np.ndarray,
            patch_shape: tuple,
            stride_shape: tuple,
            **kwargs
    ):
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

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

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
                        slice(x, x + k_x),
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
    Filter patches containing less than the `threshold` of non-zero values.

    Args:
        raw_dataset (ndarray): raw data
        label_dataset (ndarray): ground truth labels
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
        ignore_index (int): ignore index in the label dataset; this label will be matched to 0 before filtering
        threshold (float): the threshold of non-zero values in the label patch
        slack_acceptance (float): the probability of accepting a patch that does not meet the threshold criteria
        kwargs: additional metadata
    """

    def __init__(
            self,
            raw_dataset: np.ndarray,
            label_dataset: np.ndarray,
            patch_shape: tuple,
            stride_shape: tuple,
            ignore_index: Optional[int] = None,
            threshold: float = 0.6,
            slack_acceptance: float = 0.01,
            **kwargs
    ):
        super().__init__(raw_dataset, label_dataset, patch_shape, stride_shape, **kwargs)
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
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # log number of filtered patches
        logger.info(
            f"Loading {len(filtered_slices)} out of {len(self.raw_slices)} patches: "
            f"{int(100 * len(filtered_slices) / len(self.raw_slices))}%"
        )
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


def get_slice_builder(raw: np.ndarray, label: np.ndarray, config: dict) -> SliceBuilder:
    assert 'name' in config
    logger.info(f"Slice builder config: {config}")
    slice_builder_cls = _loader_classes(config['name'])
    return slice_builder_cls(raw, label, **config)


def get_train_loaders(config: dict) -> dict[str, DataLoader]:
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).
    Args:
        config:  a top level configuration object containing the 'loaders' key
    Returns:
        dict {
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
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}'
        )
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=num_workers, drop_last=True),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, pin_memory=True,
                          num_workers=num_workers, drop_last=True)
    }


def get_test_loaders(config: dict) -> DataLoader:
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
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}'
        )
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


def calculate_stats(img: np.array, skip: bool = False) -> dict[str, Any]:
    """
    Calculates the minimum percentile, maximum percentile, mean, and standard deviation of the image.

    Args:
        img: The input image array.
        skip: if True, skip the calculation and return None for all values.

    Returns:
        tuple[float, float, float, float]: The minimum percentile, maximum percentile, mean, and std dev
    """
    if not skip:
        pmin, pmax, mean, std = np.percentile(img, 1), np.percentile(img, 99.6), np.mean(img), np.std(img)
    else:
        pmin, pmax, mean, std = None, None, None, None

    return {
        'pmin': pmin,
        'pmax': pmax,
        'mean': mean,
        'std': std
    }


def mirror_pad(image, padding_shape):
    """
    Pad the image with a mirror reflection of itself.

    This function is used on data in its original shape before it is split into patches.

    Args:
        image (np.ndarray): The input image array to be padded.
        padding_shape (tuple of int): Specifies the amount of padding for each dimension, should be YX or ZYX.

    Returns:
        np.ndarray: The mirror-padded image.

    Raises:
        ValueError: If any element of padding_shape is negative.
    """
    assert len(padding_shape) == 3, "Padding shape must be specified for each dimension: ZYX"

    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    pad_width = [(p, p) for p in padding_shape]

    if image.ndim == 4:
        pad_width = [(0, 0)] + pad_width
    return np.pad(image, pad_width, mode='reflect')


def remove_padding(m, padding_shape):
    """
    Removes padding from the margins of a multi-dimensional array.

    Args:
        m (np.ndarray): The input array to be unpadded.
        padding_shape (tuple of int, optional): The amount of padding to remove from each dimension.
            Assumes the tuple length matches the array dimensions.

    Returns:
        np.ndarray: The unpadded array.
    """
    if padding_shape is None:
        return m

    # Correctly construct slice objects for each dimension in padding_shape and apply them to m.
    return m[(..., *(slice(p, -p or None) for p in padding_shape))]
