import collections
import os

import h5py
import torch

from pytorch3dunet.augment import transforms
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
        self.label_transform = transformer.label_transform()

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration

        if self.phase == 'test':
            img = self.raw_images[index]
            label = self.label_images[index]
            return self.raw_transform(img), self.label_transform(label), self.paths[index]
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.raw_images)

    @classmethod
    def prediction_collate(cls, batch):
        error_msg = "batch must contain tensors or str; found {}"
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, 0)
        elif isinstance(batch[0], str):
            return batch[0]
        elif isinstance(batch[0], collections.Sequence):
            # transpose tuples, i.e. [[1, 2], ['a', 'b']] to be [[1, 'a'], [2, 'b']]
            transposed = zip(*batch)
            return [cls.prediction_collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = phase_config['file_paths']
        return [cls(file_paths[0], phase, transformer_config)]
