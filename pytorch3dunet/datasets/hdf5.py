import glob
import os
from itertools import chain

import h5py

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('HDF5Dataset')


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

        input_file = h5py.File(file_path, 'r')

        self.raw = self._load_dataset(input_file, raw_internal_path)

        stats = calculate_stats(self.raw, global_normalization)

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()
            self.label = self._load_dataset(input_file, label_internal_path)

            if weight_internal_path is not None:
                # look for the weight map in the raw file
                self.weight_map = self._load_dataset(input_file, weight_internal_path)
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_map = None

            self._check_volume_sizes(self.raw, self.label)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None
            self.weight_map = None

        # build slice indices for raw and label data sets
        slice_builder = get_slice_builder(self.raw, self.label, self.weight_map, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    def load_dataset(self, input_file, internal_path):
        raise NotImplementedError

    def _load_dataset(self, input_file, internal_path):
        assert internal_path in input_file, f"Internal path: {internal_path} not found in the H5 file"
        ds = self.load_dataset(input_file, internal_path)
        assert ds.ndim in [3, 4], \
            f"Invalid dataset dimension: {ds.ndim}. Supported dataset formats: (C, Z, Y, X) or (Z, Y, X)"
        return ds

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(self.raw[raw_idx])

        if self.phase == 'test':
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.label[label_idx])
            if self.weight_map is not None:
                weight_idx = self.weight_slices[idx]
                weight_patch_transformed = self.weight_transform(self.weight_map[weight_idx])
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

        assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

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
        file_paths = cls.traverse_h5_paths(file_paths)

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

    @staticmethod
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

    def load_dataset(self, input_file, internal_path):
        # load the dataset from the H5 file into memory
        return input_file[internal_path][:]


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

    def load_dataset(self, input_file, internal_path):
        # load the dataset from the H5 file lazily
        return input_file[internal_path]
