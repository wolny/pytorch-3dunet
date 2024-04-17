import glob
import os
from abc import abstractmethod
from itertools import chain

import h5py

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats, mirror_pad
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('HDF5Dataset')


def _create_padded_indexes(indexes, halo_shape):
    return tuple(slice(index.start, index.stop + 2 * halo) for index, halo in zip(indexes, halo_shape))


def traverse_h5_paths(file_paths):
    assert isinstance(file_paths, list)
    results = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # if file path is a directory take all H5 files in that directory
            iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
            for fp in chain(*iters):
                results.append(fp)
        else:
            results.append(file_path)
    return results


class AbstractHDF5Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Args:
        file_path (str): path to H5 file containing raw data as well as labels and per pixel weights (optional)
        phase (str): 'train' for training, 'val' for validation, 'test' for testing
        slice_builder_config (dict): configuration of the SliceBuilder
        transformer_config (dict): data augmentation configuration
        raw_internal_path (str or list): H5 internal path to the raw dataset
        label_internal_path (str or list): H5 internal path to the label dataset
        weight_internal_path (str or list): H5 internal path to the per pixel weights (optional)
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, raw_internal_path='raw',
                 label_internal_path='label', weight_internal_path=None, global_normalization=True):
        assert phase in ['train', 'val', 'test']

        self.phase = phase
        self.file_path = file_path
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path
        self.weight_internal_path = weight_internal_path

        self.halo_shape = slice_builder_config.get('halo_shape', [0, 0, 0])

        if global_normalization:
            logger.info('Calculating mean and std of the raw data...')
            with h5py.File(file_path, 'r') as f:
                raw = f[raw_internal_path][:]
                stats = calculate_stats(raw)
        else:
            stats = calculate_stats(None, True)

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()

            if weight_internal_path is not None:
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_transform = None

            self._check_volume_sizes()
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None
            self.weight_map = None

            # compare patch and stride configuration
            patch_shape = slice_builder_config.get('patch_shape')
            stride_shape = slice_builder_config.get('stride_shape')
            if sum(self.halo_shape) != 0 and patch_shape != stride_shape:
                logger.warning(f'Found non-zero halo shape {self.halo_shape}. '
                               f'In this case: patch shape and stride shape should be equal for optimal prediction '
                               f'performance, but found patch_shape: {patch_shape} and stride_shape: {stride_shape}!')

        with h5py.File(file_path, 'r') as f:
            raw = f[raw_internal_path]
            label = f[label_internal_path] if phase != 'test' else None
            weight_map = f[weight_internal_path] if weight_internal_path is not None else None
            # build slice indices for raw and label data sets
            slice_builder = get_slice_builder(raw, label, weight_map, slice_builder_config)
            self.raw_slices = slice_builder.raw_slices
            self.label_slices = slice_builder.label_slices
            self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    @abstractmethod
    def get_raw_patch(self, idx):
        raise NotImplementedError

    @abstractmethod
    def get_label_patch(self, idx):
        raise NotImplementedError

    @abstractmethod
    def get_weight_patch(self, idx):
        raise NotImplementedError

    @abstractmethod
    def get_raw_padded_patch(self, idx):
        raise NotImplementedError

    def volume_shape(self):
        with h5py.File(self.file_path, 'r') as f:
            raw = f[self.raw_internal_path]
            if raw.ndim == 3:
                return raw.shape
            else:
                return raw.shape[1:]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]

        if self.phase == 'test':
            if len(raw_idx) == 4:
                # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                raw_idx = raw_idx[1:]  # Remove the first element if raw_idx has 4 elements
                raw_idx_padded = (slice(None),) + _create_padded_indexes(raw_idx, self.halo_shape)
            else:
                raw_idx_padded = _create_padded_indexes(raw_idx, self.halo_shape)

            raw_patch_transformed = self.raw_transform(self.get_raw_padded_patch(raw_idx_padded))
            return raw_patch_transformed, raw_idx
        else:
            raw_patch_transformed = self.raw_transform(self.get_raw_patch(raw_idx))

            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.get_label_patch(label_idx))
            if self.weight_internal_path is not None:
                weight_idx = self.weight_slices[idx]
                weight_patch_transformed = self.weight_transform(self.get_weight_patch(weight_idx))
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    def _check_volume_sizes(self):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        with h5py.File(self.file_path, 'r') as f:
            raw = f[self.raw_internal_path]
            label = f[self.label_internal_path]
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'
            if self.weight_internal_path is not None:
                weight_map = f[self.weight_internal_path]
                assert weight_map.ndim in [3, 4], 'Weight map dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
                assert _volume_shape(raw) == _volume_shape(weight_map), 'Raw and weight map have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = traverse_h5_paths(file_paths)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets


class StandardHDF5Dataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config,
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None,
                 global_normalization=True):
        super().__init__(file_path=file_path, phase=phase, slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config, raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path, weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)
        self._raw = None
        self._raw_padded = None
        self._label = None
        self._weight_map = None

    def get_raw_patch(self, idx):
        if self._raw is None:
            with h5py.File(self.file_path, 'r') as f:
                assert self.raw_internal_path in f, f'Dataset {self.raw_internal_path} not found in {self.file_path}'
                self._raw = f[self.raw_internal_path][:]
        return self._raw[idx]

    def get_label_patch(self, idx):
        if self._label is None:
            with h5py.File(self.file_path, 'r') as f:
                assert self.label_internal_path in f, f'Dataset {self.label_internal_path} not found in {self.file_path}'
                self._label = f[self.label_internal_path][:]
        return self._label[idx]

    def get_weight_patch(self, idx):
        if self._weight_map is None:
            with h5py.File(self.file_path, 'r') as f:
                assert self.weight_internal_path in f, f'Dataset {self.weight_internal_path} not found in {self.file_path}'
                self._weight_map = f[self.weight_internal_path][:]
        return self._weight_map[idx]

    def get_raw_padded_patch(self, idx):
        if self._raw_padded is None:
            with h5py.File(self.file_path, 'r') as f:
                assert self.raw_internal_path in f, f'Dataset {self.raw_internal_path} not found in {self.file_path}'
                self._raw_padded = mirror_pad(f[self.raw_internal_path][:], self.halo_shape)
        return self._raw_padded[idx]


class LazyHDF5Dataset(AbstractHDF5Dataset):
    """Implementation of the HDF5 dataset which loads the data lazily. It's slower, but has a low memory footprint."""

    def __init__(self, file_path, phase, slice_builder_config, transformer_config,
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None,
                 global_normalization=False):
        super().__init__(file_path=file_path, phase=phase, slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config, raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path, weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)

        logger.info("Using LazyHDF5Dataset")

    def get_raw_patch(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            return f[self.raw_internal_path][idx]

    def get_label_patch(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            return f[self.label_internal_path][idx]

    def get_weight_patch(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            return f[self.weight_internal_path][idx]

    def get_raw_padded_patch(self, idx):
        with h5py.File(self.file_path, 'r+') as f:
            if 'raw_padded' in f:
                return f['raw_padded'][idx]

            raw = f[self.raw_internal_path][:]
            raw_padded = mirror_pad(raw, self.halo_shape)
            f.create_dataset('raw_padded', data=raw_padded, compression='gzip')
            return raw_padded[idx]
