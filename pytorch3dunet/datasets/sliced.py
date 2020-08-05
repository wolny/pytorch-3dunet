import collections
import os

import h5py
import torch

from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.dsb import dsb_prediction_collate, DSB2018Dataset
from pytorch3dunet.datasets.utils import ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('SlicedDataset')


def _load_images(root_dir):
    raw_images = []
    label_images = []
    paths = []
    for file in os.listdir(root_dir):
        path = os.path.join(root_dir, file)
        if os.path.isdir(path):
            continue

        with h5py.File(path, 'r') as f:
            raw = f['raw'][...]
            label = f['label'][...]

        raw_images.append(raw)
        label_images.append(label)
        paths.append(path)

    return raw_images, label_images, paths


class SlicedDataset(ConfigDataset):
    def __init__(self, root_dir, phase, transformer_config):
        assert os.path.isdir(root_dir), f'{root_dir} is not a directory'
        assert phase in ['train', 'val', 'test']
        self.phase = phase

        self.file_path = root_dir

        self.raw_images, self.label_images, self.paths = _load_images(root_dir)
        min_value, max_value, mean, std = calculate_stats(self.raw_images)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')

        transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                 mean=mean, std=std)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()
        if 'label' in transformer_config:
            self.label_transform = transformer.label_transform()
        else:
            self.label_transform = None

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration

        if self.phase != 'test':
            img = self.raw_images[index]
            label = self.label_images[index]
            return self.raw_transform(img), self.label_transform(label)
        else:
            img = self.raw_images[index]
            label = self.label_images[index]
            # just a hack to get the embedding anchors easier from the target labels
            if self.label_transform is None:
                return self.raw_transform(img), self.paths[index]
            else:
                return self.raw_transform(img), self.label_transform(label), self.paths[index]

    def __len__(self):
        return len(self.raw_images)

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = phase_config['file_paths']
        assert len(file_paths) == 1, \
            f'Expected single path to the directory containing slices, but {len(file_paths)} given'
        return [cls(file_paths[0], phase, transformer_config)]


class DSBRootDataset(ConfigDataset):
    def __init__(self, root_dir, phase, transformer_config):
        self.dsb_dataset = DSB2018Dataset(root_dir[0], phase, transformer_config)
        self.sliced_dataset = SlicedDataset(root_dir[1], phase, transformer_config)

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration

        if self.dsb_dataset.phase != 'test':
            if index < len(self.dsb_dataset):
                img, label = self.dsb_dataset[index]
                domain = 0
            else:
                img, label = self.sliced_dataset[index - len(self.dsb_dataset)]
                domain = 1

            return img, label, domain
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.dsb_dataset) + len(self.sliced_dataset)

    @classmethod
    def prediction_collate(cls, batch):
        error_msg = "batch must contain tensors or str; found {}"
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, 0)
        elif isinstance(batch[0], str):
            return list(batch)
        elif isinstance(batch[0], int):
            return list(batch)
        elif isinstance(batch[0], collections.Sequence):
            # transpose tuples, i.e. [[1, 2], ['a', 'b']] to be [[1, 'a'], [2, 'b']]
            transposed = zip(*batch)
            return [dsb_prediction_collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = phase_config['file_paths']
        if phase == 'train':
            assert len(file_paths) == 2, f'Expected 2 directory paths, but {len(file_paths)} given'
            return [cls(file_paths, phase, transformer_config)]
        elif phase == 'val':
            assert len(file_paths) == 1, f'Expected 1 directory paths, but {len(file_paths)} given'
            return [DSB2018Dataset(file_paths[0], phase, transformer_config)]
        else:
            raise NotImplementedError()

