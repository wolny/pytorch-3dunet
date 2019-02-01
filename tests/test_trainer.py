import logging
import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch
import torch.optim as optim

from datasets.hdf5 import get_loaders
from unet3d.losses import DiceCoefficient, get_loss_criterion
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
        final_sigmoid = loss == 'bce'
        loss_criterion = get_loss_criterion(loss, weight=torch.rand(2).to(device))
        model = self._create_model(final_sigmoid, conv_layer_order)
        accuracy_criterion = DiceCoefficient()
        channel_per_class = loss == 'bce'
        if loss in ['bce']:
            label_dtype = 'float32'
        else:
            label_dtype = 'long'
        pixel_wise_weight = loss == 'pce'

        patch = (32, 64, 64)
        stride = (32, 64, 64)
        train, val = TestUNet3DTrainer._create_random_dataset((128, 128, 128), (64, 64, 64), channel_per_class)
        loaders = get_loaders([train], [val], 'raw', 'label', label_dtype=label_dtype, train_patch=patch,
                              train_stride=stride, val_patch=patch, val_stride=stride, transformer='BaseTransformer',
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
    def _create_model(final_sigmoid, layer_order):
        in_channels = 1
        out_channels = 2
        # use F.interpolate for upsampling and 16 initial feature maps to speed up the tests
        return UNet3D(in_channels, out_channels, init_channel_number=16, final_sigmoid=final_sigmoid, interpolate=True,
                      conv_layer_order=layer_order)

    @staticmethod
    def _create_random_dataset(train_shape, val_shape, channel_per_class):
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
                f.create_dataset('weight_map', data=np.random.rand(*shape))

            result.append(tmp.name)

        return result
