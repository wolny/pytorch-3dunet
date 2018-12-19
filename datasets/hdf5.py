import h5py
import numpy as np
from torch.utils.data import Dataset

import augment.transforms as transforms


class HDF5Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the H5(files), which iterates over the raw and label datasets
    patch by patch with a given stride.
    If one would like to add on the fly data augmentation to the patches they should subclass and override
    the 'get_transforms()' method.
    """

    def __init__(self, raw_file_path, patch_shape, stride_shape, phase, label_file_path=None, raw_internal_path='raw',
                 label_internal_path='label', label_dtype=np.long, transformer=transforms.BaseTransformer, **kwargs):
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
        :param kwargs: additional context parameters
        """
        assert phase in ['train', 'val', 'test']
        self._check_patch_shape(patch_shape)
        self.phase = phase
        self.label_dtype = label_dtype

        self.raw_file = h5py.File(raw_file_path, 'r')
        self.raw = self.raw_file[raw_internal_path]
        # build index->slice mapping
        self.raw_slices = self._build_slices(self.raw.shape, patch_shape, stride_shape)

        # create raw and label transforms
        mean, std = self.calculate_mean_std()
        if 'angle_spectrum' in kwargs:
            angle_spectrum = kwargs['angle_spectrum']
        else:
            angle_spectrum = 10

        self.transformer = transformer.create(mean=mean, std=std, phase=phase, label_dtype=label_dtype,
                                              angle_spectrum=angle_spectrum)

        self.raw_transform = self.transformer.raw_transform()
        self.label_transform = self.transformer.label_transform()

        # 'test' phase used only for predictions so ignore the label dataset
        if phase != 'test':
            # if label_file_path is None assume that labels are stored in the raw_file_path as well
            if label_file_path is None:
                self.label_file = self.raw_file
            else:
                self.label_file = h5py.File(label_file_path, 'r')

            self.label = self.label_file[label_internal_path]
            self._check_dimensionality(self.raw, self.label)
            self.label_slices = self._build_slices(self.label.shape, patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self.label_slices)
        else:
            self.label = None

        self.patch_count = len(self.raw_slices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]
        img_slice_tensor = self.raw_transform(self.raw[raw_idx])

        if self.phase != 'test':
            label_idx = self.label_slices[idx]
            label_slice_tensor = self.label_transform(self.label[label_idx])
            return img_slice_tensor, label_slice_tensor
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
    def _build_slices(shape, patch_shape, stride_shape):
        """Iterates over the 3-dimensional array patch-by-patch with a given stride
        and builds the mapping from index to slice position.

        Args:
            shape (tuple): shape of the n-dim array
            patch_shape (tuple): patch shape
            stride_shape (tuple): stride shape

        Returns:
            index to slice mapping
            (int -> (slice, slice, slice, slice)) if len(shape) == 4
            (int -> (slice, slice, slice)) if len(shape) == 3
        """
        slices = {}
        if len(shape) == 4:
            in_channels, i_z, i_y, i_x = shape
        else:
            i_z, i_y, i_x = shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        idx = 0
        z_steps = HDF5Dataset._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = HDF5Dataset._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = HDF5Dataset._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if len(shape) == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices[idx] = slice_idx
                    idx += 1
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

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


class WeightedHDF5Dataset(HDF5Dataset):
    def __init__(self, raw_file_path, patch_shape, stride_shape, phase, label_file_path=None, raw_internal_path='raw',
                 label_internal_path='label', label_dtype=np.long, **kwargs):
        super().__init__(raw_file_path, patch_shape, stride_shape, phase, label_file_path, raw_internal_path,
                         label_internal_path, label_dtype, transforms.StandardTransformerWithWeights, **kwargs)
        self.weight_map = self.raw_file['weight_map']
        self.weight_transform = self.transformer.get_weight_transform()

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]
        img_slice_tensor = self.raw_transform(self.raw[raw_idx])
        weight_slice_tensor = self.weight_transform(self.weight_map[raw_idx])

        if self.phase != 'test':
            label_idx = self.label_slices[idx]
            label_slice_tensor = self.label_transform(self.label[label_idx])
            return img_slice_tensor, label_slice_tensor, weight_slice_tensor
        else:
            # if in the 'test' phase return the slice metadata as well
            return img_slice_tensor, raw_idx
