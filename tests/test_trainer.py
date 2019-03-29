import logging
import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from datasets.hdf5 import get_train_loaders
from unet3d.losses import get_loss_criterion
from unet3d.metrics import get_evaluation_metric
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger

CONFIG_BASE = {
    'loaders': {
        'train_patch': [32, 64, 64],
        'train_stride': [32, 64, 64],
        'val_patch': [32, 64, 64],
        'val_stride': [32, 64, 64],
        'raw_internal_path': 'raw',
        'label_internal_path': 'label',
        'weight_internal_path': None,
        'transformer': {
            'train': {
                'raw': [{'name': 'Normalize'}, {'name': 'ToTensor', 'expand_dims': True}],
                'label': [{'name': 'ToTensor', 'expand_dims': False}],
                'weight': [{'name': 'ToTensor', 'expand_dims': False}]
            },
            'test': {
                'raw': [{'name': 'Normalize'}, {'name': 'ToTensor', 'expand_dims': True}],
                'label': [{'name': 'ToTensor', 'expand_dims': False}],
                'weight': [{'name': 'ToTensor', 'expand_dims': False}]
            }
        }
    }
}


class TestUNet3DTrainer:
    def test_ce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'ce', 'iou')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_wce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'wce', 'iou')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_bce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'bce', 'dice')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_dice_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'dice', 'iou')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_pce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'pce', 'iou', weight_map=True)

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def _train_save_load(self, tmpdir, loss, val_metric, max_num_epochs=1, log_after_iters=2, validate_after_iters=2,
                         max_num_iterations=4, weight_map=False):
        # conv-relu-groupnorm
        conv_layer_order = 'crg'
        final_sigmoid = loss in ['bce', 'dice', 'gdl']
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        test_config = dict(CONFIG_BASE)
        test_config.update({
            # get device to train on
            'device': device,
            'loss': {'name': loss, 'weight': np.random.rand(2).astype(np.float32)},
            'eval_metric': {'name': val_metric}
        })
        if weight_map:
            test_config['loaders']['weight_internal_path'] = 'weight_map'

        loss_criterion = get_loss_criterion(test_config)
        eval_criterion = get_evaluation_metric(test_config)
        model = self._create_model(final_sigmoid, conv_layer_order)
        channel_per_class = loss in ['bce', 'dice', 'gdl']
        if loss in ['bce']:
            label_dtype = 'float32'
        else:
            label_dtype = 'long'
        test_config['loaders']['transformer']['train']['label'][0]['dtype'] = label_dtype
        test_config['loaders']['transformer']['test']['label'][0]['dtype'] = label_dtype

        train, val = TestUNet3DTrainer._create_random_dataset((128, 128, 128), (64, 64, 64), channel_per_class)
        test_config['loaders']['train_path'] = [train]
        test_config['loaders']['val_path'] = [val]

        loaders = get_train_loaders(test_config)

        learning_rate = 2e-4
        weight_decay = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = MultiStepLR(optimizer, milestones=[2, 3], gamma=0.5)
        logger = get_logger('UNet3DTrainer', logging.DEBUG)
        trainer = UNet3DTrainer(model, optimizer, lr_scheduler,
                                loss_criterion, eval_criterion,
                                device, loaders, tmpdir,
                                max_num_epochs=max_num_epochs,
                                log_after_iters=log_after_iters,
                                validate_after_iters=validate_after_iters,
                                max_num_iterations=max_num_iterations,
                                logger=logger)
        trainer.fit()
        # test loading the trainer from the checkpoint
        trainer = UNet3DTrainer.from_checkpoint(os.path.join(tmpdir, 'last_checkpoint.pytorch'),
                                                model, optimizer, lr_scheduler,
                                                loss_criterion, eval_criterion,
                                                loaders, logger=logger)
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
