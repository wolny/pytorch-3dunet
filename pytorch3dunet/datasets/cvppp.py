import os
import random

import numpy as np
from PIL import Image
from torchvision import transforms as ts

from pytorch3dunet.augment.transforms import Relabel
from pytorch3dunet.datasets.dsb import dsb_prediction_collate
from pytorch3dunet.datasets.utils import ConfigDataset, cvppp_sample_instances, RgbToLabel, \
    LabelToTensor
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('CVPPP2017Dataset')


class CVPPP2017Dataset(ConfigDataset):
    def __init__(self, root_dir, phase, instance_ratio=None):
        assert os.path.isdir(root_dir), f'{root_dir} is not a directory'
        assert phase in ['train', 'val', 'test']

        self.phase = phase

        self.images, self.paths = self._load_files(root_dir, suffix='rgb')
        self.file_path = root_dir
        self.instance_ratio = instance_ratio

        self.raw_transform = ts.Compose(
            [
                ts.RandomHorizontalFlip(),
                ts.RandomVerticalFlip(),
                ts.RandomResizedCrop(448, scale=(0.7, 1.)),
                ts.ToTensor(),
                ts.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
                # add singleton z-dim
                lambda m: m.unsqueeze(1)
            ]
        )

        self.train_label_transform = ts.Compose(
            [
                ts.RandomHorizontalFlip(),
                ts.RandomVerticalFlip(),
                ts.RandomResizedCrop(448, scale=(0.7, 1.), interpolation=0),
                RgbToLabel(),
                Relabel(run_cc=False),
                LabelToTensor(),
                lambda m: m.unsqueeze(0)
            ]
        )

        self.val_label_transform = ts.Compose(
            [
                ts.Resize(size=(448, 448)),
                LabelToTensor(),
                lambda m: m.unsqueeze(0)
            ]
        )

        self.test_transform = ts.Compose(
            [
                ts.Resize(size=(448, 448)),
                ts.ToTensor(),
                ts.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
                # add singleton z-dim
                lambda m: m.unsqueeze(1)
            ]
        )

        if phase == 'train':
            # load labeled images
            self.masks, _ = self._load_files(root_dir, 'label')
            # training with sparse object supervision
            if self.instance_ratio is not None and phase == 'train':
                assert 0 < self.instance_ratio <= 1
                rs = np.random.RandomState(47)
                self.masks = [cvppp_sample_instances(m, self.instance_ratio, rs) for m in self.masks]

            assert len(self.images) == len(self.masks)
        elif phase == 'val':
            # load labeled images
            self.masks, _ = self._load_files(root_dir, 'fg')
            assert len(self.images) == len(self.masks)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase == 'train':
            mask = self.masks[idx]
            seed = np.random.randint(np.iinfo('int32').max)
            random.seed(seed)
            img = self.raw_transform(img)
            random.seed(seed)
            mask = self.train_label_transform(mask)
            return img, mask
        elif self.phase == 'val':
            mask = self.masks[idx]
            seed = np.random.randint(np.iinfo('int32').max)
            random.seed(seed)
            img = self.test_transform(img)
            random.seed(seed)
            mask = self.val_label_transform(mask)
            return img, mask
        else:
            return self.test_transform(img), self.paths[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load files to process
        file_paths = phase_config['file_paths']
        instance_ratio = phase_config.get('instance_ratio', None)
        return [cls(file_paths[0], phase, instance_ratio)]

    @staticmethod
    def _load_files(dir, suffix):
        # we only load raw or label images
        assert suffix in ['rgb', 'label', 'fg']
        files_data = []
        paths = []
        for file in sorted(os.listdir(dir)):
            base = os.path.splitext(file)[0]
            if base.endswith(suffix):
                path = os.path.join(dir, file)
                # load image
                img = Image.open(path)
                if suffix in ['rgb', 'label']:
                    img = img.convert('RGB')
                # save image and path
                files_data.append(img)
                paths.append(path)

        return files_data, paths
