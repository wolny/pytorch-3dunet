import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import augment.transforms as transforms


class SliceBuilder:
    def __init__(self, raw_dataset, label_datasets, patch_shape, stride_shape):
        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_datasets is None:
            self._label_slices = None
        else:
            # take the first element in the label_datasets to build slices
            # TODO: won't work for curriculum slice builder
            self._label_slices = self._build_slices(label_datasets[0], patch_shape, stride_shape)
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


class CurriculumLearningSliceBuilder(SliceBuilder):
    """
    Simple Curriculum Learning strategy when we show patches with less volume of 'ignore_index' (label patch) first.
    The hypothesis is that having less 'ignore_index' patches at the beginning of the epoch will lead to faster
    convergence and better local minima.
    """

    def __init__(self, raw_dataset, label_datasets, patch_shape, stride_shape, ignore_index=-1):
        super().__init__(raw_dataset, label_datasets, patch_shape, stride_shape)
        if label_datasets is None:
            return

        def ignore_index_volume(raw_label_idx):
            _, label_idx = raw_label_idx
            label_patch = label_datasets[0][label_idx]
            return np.count_nonzero(label_patch == ignore_index)

        # after computing the patches sort them so that patches with less ignore_index volume come first
        zipped_slices = zip(self.raw_slices, self.label_slices)
        sorted_slices = sorted(zipped_slices, key=ignore_index_volume)
        # unzip as save
        raw_slices, label_slices = zip(*sorted_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


class HDF5Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path, patch_shape, stride_shape, phase, transformer_config,
                 raw_internal_path='raw', label_internal_path='label',
                 weight_internal_path=None, slice_builder_cls=SliceBuilder):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param transformer_config: data augmentation configuration
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        :param weight_internal_path (str or list): H5 internal path to the per pixel weights
        :param slice_builder_cls: defines how to sample the patches from the volume
        """
        assert phase in ['train', 'val', 'test']
        self._check_patch_shape(patch_shape)
        self.phase = phase
        self.file_path = file_path
        self.input_file = h5py.File(file_path, 'r')
        self.raw = self.input_file[raw_internal_path]

        # create raw and label transforms
        mean, std = self._calculate_mean_std(self.raw[...])

        self.transformer = transforms.get_transformer(transformer_config, mean, std, phase)
        self.raw_transform = self.transformer.raw_transform()

        # convert label_internal_path and weight_internal_path to list for ease of computation
        if isinstance(label_internal_path, str):
            label_internal_path = [label_internal_path]
        if isinstance(weight_internal_path, str):
            weight_internal_path = [weight_internal_path]

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()
            self.labels = [self.input_file[internal_path] for internal_path in label_internal_path]

            if weight_internal_path is not None:
                # look for the weight map in the raw file
                self.weight_maps = [self.input_file[internal_path] for internal_path in weight_internal_path]
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_maps = None

            self._check_dimensionality(self.raw, self.labels)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.labels = None

        # build slice indices for raw and label data sets
        slice_builder = slice_builder_cls(self.raw, self.labels, patch_shape, stride_shape)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        self.patch_count = len(self.raw_slices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch = self.raw[raw_idx]
        # apply the raw transformer
        raw_patch_transformed = self.raw_transform(raw_patch)

        if self.phase == 'test':
            # just return the transformed raw patch and the metadata
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self._transform_patches(self.labels, label_idx, self.label_transform)
            if self.weight_maps is not None:
                # return the transformed voxel weight map for a given patch together with raw and label data
                weight_patch_transformed = self._transform_patches(self.weight_maps, raw_idx, self.weight_transform)
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the label data and apply the label transformer
            transformed_patch = transformer(dataset[label_idx])
            transformed_patches.append(transformed_patch)

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches

    def __len__(self):
        return self.patch_count

    def close(self):
        self.input_file.close()

    @staticmethod
    def _calculate_mean_std(input):
        """
        Compute a channel-wise mean/std of the raw stack for normalization.
        This is an in-memory implementation, override this method
        with the chunk-based computation if you're working with huge H5 files.
        :return: a tuple of (mean, std) of the raw data
        """
        return input.mean(keepdims=True), input.std(keepdims=True)

    @staticmethod
    def _check_dimensionality(raw, labels):
        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

        if raw.ndim == 3:
            raw_shape = raw.shape
        else:
            raw_shape = raw.shape[1:]

        for label in labels:
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            if label.ndim == 3:
                label_shape = label.shape
            else:
                label_shape = label.shape[1:]
            assert raw_shape == label_shape, 'Raw and labels have to be of the same size'

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'
        assert patch_shape[0] >= 16, 'Depth must be greater or equal 16'


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders
    (torch.utils.data.DataLoader) backed by the datasets.hdf5.HDF5Dataset.

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    # get train and validation files
    train_paths = loaders_config['train_path']
    val_paths = loaders_config['val_path']
    assert isinstance(train_paths, list)
    assert isinstance(val_paths, list)
    # get h5 internal paths for raw and label
    raw_internal_path = loaders_config['raw_internal_path']
    label_internal_path = loaders_config['label_internal_path']
    weight_internal_path = loaders_config.get('weight_internal_path', None)
    # get train/validation patch size and stride
    train_patch = tuple(loaders_config['train_patch'])
    train_stride = tuple(loaders_config['train_stride'])
    val_patch = tuple(loaders_config['val_patch'])
    val_stride = tuple(loaders_config['val_stride'])

    curriculum_sampler = loaders_config.get('curriculum', False)
    if curriculum_sampler:
        slice_builder_cls = CurriculumLearningSliceBuilder
    else:
        slice_builder_cls = SliceBuilder

    train_datasets = []
    for train_path in train_paths:
        # create H5 backed training and validation dataset with data augmentation
        train_dataset = HDF5Dataset(train_path, train_patch, train_stride, phase='train',
                                    transformer_config=loaders_config['transformer'],
                                    raw_internal_path=raw_internal_path,
                                    label_internal_path=label_internal_path,
                                    weight_internal_path=weight_internal_path,
                                    slice_builder_cls=slice_builder_cls)
        train_datasets.append(train_dataset)

    val_datasets = []
    for val_path in val_paths:
        val_dataset = HDF5Dataset(val_path, val_patch, val_stride, phase='val',
                                  transformer_config=loaders_config['transformer'],
                                  raw_internal_path=raw_internal_path,
                                  label_internal_path=label_internal_path,
                                  weight_internal_path=weight_internal_path)
        val_datasets.append(val_dataset)

    # shuffle only if curriculum learning scheme is not used
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=1, shuffle=not curriculum_sampler),
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=1, shuffle=not curriculum_sampler)
    }


def get_test_datasets(config):
    """
    Returns a list of HDF5Datasets, one per each test file.

    :param config: a top level configuration object containing the 'datasets' key
    :return: list of HDF5Dataset objects
    """
    assert 'datasets' in config, 'Could not find data sets configuration'
    datasets_config = config['datasets']

    # get train and validation files
    test_paths = datasets_config['test_path']
    assert isinstance(test_paths, list)
    # get h5 internal paths for raw and label
    raw_internal_path = datasets_config['raw_internal_path']
    # get train/validation patch size and stride
    patch = tuple(datasets_config['patch'])
    stride = tuple(datasets_config['stride'])
    # return HDF5Dataset per test path
    return [HDF5Dataset(test_path, patch, stride, phase='test', raw_internal_path=raw_internal_path,
                        transformer_config=datasets_config['transformer']) for test_path in test_paths]
