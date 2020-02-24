import os

import imageio
import numpy as np

from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('DSB2018Dataset')


class DSB2018Dataset(ConfigDataset):
    def __init__(self, root_dir, phase, transformer_config, mirror_padding=(0, 32, 32), expand_dims=True):
        assert os.path.isdir(root_dir), 'root_dir is not a directory'
        assert phase in ['train', 'val', 'test']

        # use mirror padding only during the 'test' phase
        if phase in ['train', 'val']:
            mirror_padding = None
        if mirror_padding is not None:
            assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"
        self.mirror_padding = mirror_padding

        self.phase = phase

        # load raw images
        images_dir = os.path.join(root_dir, 'images')
        assert os.path.isdir(images_dir)
        self.images = self._load_files(images_dir, expand_dims)

        min_value, max_value, mean, std = calculate_stats(self.images)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')

        transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                 mean=mean, std=std)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != 'test':
            # load labeled images
            masks_dir = os.path.join(root_dir, 'masks')
            assert os.path.isdir(masks_dir)
            self.masks = self._load_files(masks_dir, expand_dims)
            assert len(self.images) == len(self.masks)
            # load label images transformer
            self.masks_transform = transformer.label_transform()
        else:
            self.masks = None
            self.masks_transform = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                padded_imgs = []
                for img in self.images:
                    padded_img = np.pad(img, pad_width=pad_width, mode='reflect')
                    padded_imgs.append(padded_img)

                self.images = padded_imgs

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase != 'test':
            mask = self.masks[idx]
            return self.raw_transform(img), self.masks_transform(mask)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.images)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = phase_config['file_paths']
        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)

        if phase != 'test':
            return [cls(file_paths[0], phase, transformer_config, mirror_padding)]
        else:
            raise NotImplementedError

    @staticmethod
    def _load_files(dir, expand_dims):
        files_data = []
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            img = np.asarray(imageio.imread(path))
            if expand_dims:
                img = np.expand_dims(img, axis=0)

            files_data.append(img)

        return files_data
