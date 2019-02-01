import importlib

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import augment.transforms as transforms


class SliceBuilder:
    def __init__(self, raw_dataset, label_dataset, patch_shape, stride_shape):
        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
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

    def __init__(self, raw_dataset, label_dataset, patch_shape, stride_shape, ignore_index=-1):
        super().__init__(raw_dataset, label_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            return

        def ignore_index_volume(raw_label_idx):
            _, label_idx = raw_label_idx
            label_patch = label_dataset[label_idx]
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
    Implementation of torch.utils.data.Dataset backed by the H5(files), which iterates over the raw and label datasets
    patch by patch with a given stride.
    If one would like to add on the fly data augmentation to the patches they should subclass and override
    the 'get_transforms()' method.
    """
    DEFAULT_TRANSFORMER_BUILDER = transforms.TransformerBuilder(transforms.BaseTransformer,
                                                                {
                                                                    'label_dtype': 'long',
                                                                    'angle_spectrum': 10
                                                                })

    def __init__(self, raw_file_path, patch_shape, stride_shape, phase,
                 transformer_builder=DEFAULT_TRANSFORMER_BUILDER,
                 label_file_path=None, raw_internal_path='raw', label_internal_path='label',
                 slice_builder_cls=SliceBuilder, pixel_wise_weight=False):
        """
        Creates transformers for raw and label datasets and builds the index to slice mapping for raw and label datasets.
        :param raw_file_path: path to H5 file containing raw data
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
        only during the 'train' phase
        :param label_file_path: path to the H5 file containing label data or 'None' if the labels are stored in the raw
        H5 file
        :param raw_internal_path: H5 internal path to the raw dataset
        :param label_internal_path: H5 internal path to the label dataset
        """
        assert phase in ['train', 'val', 'test']
        self._check_patch_shape(patch_shape)
        self.phase = phase

        self.raw_file = h5py.File(raw_file_path, 'r')
        self.raw = self.raw_file[raw_internal_path]

        # create raw and label transforms
        mean, std = self.calculate_mean_std()
        transformer_builder.mean = mean
        transformer_builder.std = std
        transformer_builder.phase = phase

        self.transformer = transformer_builder.build()

        self.raw_transform = self.transformer.raw_transform()
        self.label_transform = self.transformer.label_transform()

        if pixel_wise_weight:
            # look for the weight map in the raw file
            self.weight_map = self.raw_file['weight_map']
            self.weight_transform = self.transformer.weight_transform()
        else:
            self.weight_map = None

        # 'test' phase used only for predictions so ignore the label dataset
        if phase != 'test':
            # if label_file_path is None assume that labels are stored in the raw_file_path as well
            if label_file_path is None:
                self.label_file = self.raw_file
            else:
                self.label_file = h5py.File(label_file_path, 'r')

            self.label = self.label_file[label_internal_path]
            self._check_dimensionality(self.raw, self.label)
        else:
            self.label = None

        slice_builder = slice_builder_cls(self.raw, self.label, patch_shape, stride_shape)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        self.patch_count = len(self.raw_slices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]
        img_slice_tensor = self.raw_transform(self.raw[raw_idx])

        if self.phase != 'test':
            label_idx = self.label_slices[idx]
            label_slice_tensor = self.label_transform(self.label[label_idx])
            if self.weight_map is None:
                return img_slice_tensor, label_slice_tensor
            else:
                # return voxel weight map for a given patch together with raw and label data
                weight_slice_tensor = self.weight_transform(self.weight_map[raw_idx])
                return img_slice_tensor, label_slice_tensor, weight_slice_tensor
        else:
            # if in the 'test' phase return the slice metadata as well
            return img_slice_tensor, raw_idx

    def __len__(self):
        return self.patch_count

    def close(self):
        self.raw_file.close()
        if self.raw_file != self.label_file:
            self.label_file.close()

    def calculate_mean_std(self):
        """
        Compute a channel-wise mean/std of the raw stack for normalization.
        This is an in-memory implementation, override this method
        with the chunk-based computation if you're working with huge H5 files.
        :return: a tuple of (mean, std) of the raw data
        """

        return self.raw[...].mean(keepdims=True), self.raw[...].std(keepdims=True)

    @staticmethod
    def _check_dimensionality(raw, label):
        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        if raw.ndim == 3:
            raw_shape = raw.shape
        else:
            raw_shape = raw.shape[1:]

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


def get_loaders(train_paths, val_paths, raw_internal_path, label_internal_path, label_dtype, train_patch, train_stride,
                val_patch, val_stride, transformer, pixel_wise_weight=False, curriculum_learning=False,
                ignore_index=None):
    """
    Returns dictionary containing the  training and validation loaders
    (torch.utils.data.DataLoader) backed by the datasets.hdf5.HDF5Dataset

    :param train_paths: paths to the H5 file containing the training set
    :param val_paths: paths to the H5 file containing the validation set
    :param raw_internal_path:
    :param label_internal_path:
    :param label_dtype: target type of the label dataset
    :param train_patch:
    :param train_stride:
    :param val_path:
    :param val_stride:
    :param transformer:
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert isinstance(train_paths, list)
    assert isinstance(val_paths, list)

    def _transformer_class(class_name):
        m = importlib.import_module('augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    if isinstance(transformer, str):
        transformer_class = _transformer_class(transformer)
        config = {'label_dtype': label_dtype, 'angle_spectrum': 10}
        if ignore_index is not None:
            config['ignore_index'] = ignore_index
        transformer_builder = transforms.TransformerBuilder(transformer_class, config)
    elif isinstance(transformer, dict):
        transformer_class = _transformer_class(transformer['name'])
        transformer['label_dtype'] = label_dtype
        if ignore_index is not None:
            transformer['ignore_index'] = ignore_index
        transformer_builder = transforms.TransformerBuilder(transformer_class, transformer)
    else:
        raise ValueError("Unsupported 'transformer' type. Can be either str or dict")

    if curriculum_learning:
        slice_builder_cls = CurriculumLearningSliceBuilder
    else:
        slice_builder_cls = SliceBuilder

    train_datasets = []
    for train_path in train_paths:
        # create H5 backed training and validation dataset with data augmentation
        train_dataset = HDF5Dataset(train_path, train_patch, train_stride, phase='train',
                                    transformer_builder=transformer_builder, raw_internal_path=raw_internal_path,
                                    label_internal_path=label_internal_path, slice_builder_cls=slice_builder_cls,
                                    pixel_wise_weight=pixel_wise_weight)
        train_datasets.append(train_dataset)

    val_datasets = []
    for val_path in val_paths:
        val_dataset = HDF5Dataset(val_path, val_patch, val_stride, phase='val',
                                  transformer_builder=transformer_builder, raw_internal_path=raw_internal_path,
                                  label_internal_path=label_internal_path, pixel_wise_weight=pixel_wise_weight)
        val_datasets.append(val_dataset)

    # shuffle only if curriculum_learning scheme is not used
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=1, shuffle=not curriculum_learning),
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=1, shuffle=not curriculum_learning)
    }
