import logging
import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.hdf5 import HDF5Dataset
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import DiceCoefficient
from unet3d.utils import get_logger


class TestUNet3DTrainer:
    def test_single_epoch(self, tmpdir, capsys):
        with capsys.disabled():
            # get device to train on
            device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

            # conv-relu-groupnorm
            conv_layer_order = 'crg'

            loss_criterion, final_sigmoid = nn.NLLLoss(), False

            model = self._create_model(final_sigmoid, conv_layer_order)

            error_criterion = DiceCoefficient()

            loaders = self._get_loaders()

            learning_rate = 2e-4
            weight_decay = 0.0001
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            logger = get_logger('UNet3DTrainer', logging.DEBUG)
            trainer = UNet3DTrainer(model, optimizer, loss_criterion,
                                    error_criterion,
                                    device, loaders, tmpdir,
                                    max_num_epochs=1,
                                    log_after_iters=4,
                                    validate_after_iters=4,
                                    max_num_iterations=16,
                                    logger=logger)

            trainer.fit()

            # test loading the trainer from the checkpoint
            UNet3DTrainer.from_checkpoint(
                os.path.join(tmpdir, 'last_checkpoint.pytorch'),
                model, optimizer, loss_criterion, error_criterion, loaders,
                logger=logger)

    @staticmethod
    def _create_model(final_sigmoid, layer_order):
        in_channels = 1
        out_channels = 2
        # use F.interpolate for upsampling
        return UNet3D(in_channels, out_channels, final_sigmoid=final_sigmoid, interpolate=True,
                      conv_layer_order=layer_order)

    @staticmethod
    def _create_random_dataset():
        result = []

        for phase in ['train', 'val']:
            tmp = NamedTemporaryFile(delete=False)

            with h5py.File(tmp.name, 'w') as f:
                if phase == 'train':
                    size = (128, 128, 128)
                else:
                    size = (64, 64, 64)
                f.create_dataset('raw', data=np.random.rand(*size))
                f.create_dataset('label', data=np.random.randint(0, 2, size))

            result.append(tmp.name)

        return result

    @staticmethod
    def _get_loaders():
        train, val = TestUNet3DTrainer._create_random_dataset()
        train_dataset = HDF5Dataset(train, patch_shape=(32, 64, 64), stride_shape=(16, 32, 32), phase='train')
        val_dataset = HDF5Dataset(val, patch_shape=(64, 64, 64), stride_shape=(64, 64, 64), phase='val')

        return {
            'train': DataLoader(train_dataset, batch_size=1, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=1, shuffle=True)
        }
