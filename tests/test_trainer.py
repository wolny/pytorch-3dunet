import logging
import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.hdf5 import HDF5Dataset, WeightedHDF5Dataset
from unet3d.losses import DiceCoefficient, WeightedCrossEntropyLoss, GeneralizedDiceLoss, PixelWiseCrossEntropyLoss
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger


class TestUNet3DTrainer:
    def test_ce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'ce')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_wce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'wce')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_bce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'bce')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_dice_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'dice')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    # @pytest.mark.skip
    def test_pce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'pce')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def _train_save_load(self, tmpdir, loss, max_num_epochs=1, log_after_iters=2, validate_after_iters=2,
                         max_num_iterations=4):
        # get device to train on
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        # conv-relu-groupnorm
        conv_layer_order = 'crg'
        loss_criterion, final_sigmoid = self._get_loss_criterion(loss, weight=torch.rand(2).to(device))
        model = self._create_model(final_sigmoid, conv_layer_order)
        accuracy_criterion = DiceCoefficient()
        channel_per_class = loss == 'bce'
        if loss in ['bce', 'dice']:
            label_dtype = 'float32'
        else:
            label_dtype = 'long'
        pixel_wise_weight = loss == 'pce'
        loaders = self._get_loaders(channel_per_class=channel_per_class, label_dtype=label_dtype,
                                    pixel_wise_weight=pixel_wise_weight)
        learning_rate = 2e-4
        weight_decay = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        logger = get_logger('UNet3DTrainer', logging.DEBUG)
        trainer = UNet3DTrainer(model, optimizer, loss_criterion,
                                accuracy_criterion,
                                device, loaders, tmpdir,
                                max_num_epochs=max_num_epochs,
                                log_after_iters=log_after_iters,
                                validate_after_iters=validate_after_iters,
                                max_num_iterations=max_num_iterations,
                                logger=logger)
        trainer.fit()
        # test loading the trainer from the checkpoint
        trainer = UNet3DTrainer.from_checkpoint(
            os.path.join(tmpdir, 'last_checkpoint.pytorch'),
            model, optimizer, loss_criterion, accuracy_criterion, loaders,
            logger=logger)
        return trainer

    @staticmethod
    def _get_loss_criterion(loss, weight):
        if loss == 'bce':
            return nn.BCELoss(), True
        elif loss == 'ce':
            return nn.CrossEntropyLoss(weight=weight), False
        elif loss == 'wce':
            return WeightedCrossEntropyLoss(weight=weight), False
        elif loss == 'pce':
            return PixelWiseCrossEntropyLoss(class_weights=weight), False
        else:
            return GeneralizedDiceLoss(weight=weight), True

    @staticmethod
    def _create_model(final_sigmoid, layer_order):
        in_channels = 1
        out_channels = 2
        # use F.interpolate for upsampling
        return UNet3D(in_channels, out_channels, final_sigmoid=final_sigmoid, interpolate=True,
                      conv_layer_order=layer_order)

    @staticmethod
    def _create_random_dataset(train_shape, val_shape, channel_per_class, pixel_wise_weight):
        result = []

        for shape in [train_shape, val_shape]:
            tmp = NamedTemporaryFile(delete=False)

            with h5py.File(tmp.name, 'w') as f:
                if channel_per_class:
                    l_shape = (2,) + shape
                else:
                    l_shape = shape
                f.create_dataset('raw', data=np.random.rand(*shape))
                f.create_dataset('label', data=np.random.randint(0, 2, l_shape))
                if pixel_wise_weight:
                    f.create_dataset('weight_map', data=np.random.rand(*shape))

            result.append(tmp.name)

        return result

    @staticmethod
    def _get_loaders(channel_per_class, label_dtype, pixel_wise_weight=False):
        train, val = TestUNet3DTrainer._create_random_dataset((128, 128, 128), (64, 64, 64), channel_per_class,
                                                              pixel_wise_weight)
        if not pixel_wise_weight:
            train_dataset = HDF5Dataset(train, patch_shape=(32, 64, 64), stride_shape=(16, 32, 32), phase='train',
                                        label_dtype=label_dtype)
            val_dataset = HDF5Dataset(val, patch_shape=(64, 64, 64), stride_shape=(64, 64, 64), phase='val',
                                      label_dtype=label_dtype)
        else:
            train_dataset = WeightedHDF5Dataset(train, patch_shape=(32, 64, 64), stride_shape=(16, 32, 32),
                                                phase='train',
                                                label_dtype=label_dtype)
            val_dataset = WeightedHDF5Dataset(val, patch_shape=(32, 64, 64), stride_shape=(16, 32, 32),
                                              phase='val',
                                              label_dtype=label_dtype)

        return {
            'train': DataLoader(train_dataset, batch_size=1, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=1, shuffle=True)
        }
