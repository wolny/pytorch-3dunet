import copy
import logging
import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch

from pytorch3dunet.datasets.hdf5 import get_train_loaders
from pytorch3dunet.train import _create_optimizer, _create_lr_scheduler
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.trainer import UNet3DTrainer
from pytorch3dunet.unet3d.utils import get_logger, DefaultTensorboardFormatter

CONFIG_BASE = {
    'model': {
        'name': 'UNet3D',
        'in_channels': 3,
        'out_channels': 2,
        'f_maps': 16
    },
    'optimizer': {
        'learning_rate': 0.0002,
        'weight_decay': 0.0001
    },
    'lr_scheduler': {
        'name': 'MultiStepLR',
        'milestones': [2, 3],
        'gamma': 0.5
    },
    'loaders': {
        'raw_internal_path': 'raw',
        'label_internal_path': 'label',
        'weight_internal_path': None,

        'train': {
            'slice_builder': {
                'name': 'SliceBuilder',
                'patch_shape': (32, 64, 64),
                'stride_shape': (32, 64, 64)
            },
            'transformer': {
                'raw': [{'name': 'Normalize'}, {'name': 'ToTensor', 'expand_dims': True}],
                'label': [{'name': 'ToTensor', 'expand_dims': False}],
                'weight': [{'name': 'ToTensor', 'expand_dims': False}]
            }
        },
        'val': {
            'slice_builder': {
                'name': 'SliceBuilder',
                'patch_shape': (32, 64, 64),
                'stride_shape': (32, 64, 64)
            },
            'transformer': {
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
            trainer = self._train_save_load(tmpdir, 'CrossEntropyLoss', 'MeanIoU')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_wce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'WeightedCrossEntropyLoss', 'MeanIoU')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_bce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'BCEWithLogitsLoss', 'DiceCoefficient')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_dice_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'DiceLoss', 'MeanIoU')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_pce_loss(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'PixelWiseCrossEntropyLoss', 'MeanIoU', weight_map=True)

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def test_residual_unet(self, tmpdir, capsys):
        with capsys.disabled():
            trainer = self._train_save_load(tmpdir, 'CrossEntropyLoss', 'MeanIoU', model='ResidualUNet3D')

            assert trainer.max_num_epochs == 1
            assert trainer.log_after_iters == 2
            assert trainer.validate_after_iters == 2
            assert trainer.max_num_iterations == 4

    def _train_save_load(self, tmpdir, loss, val_metric, model='UNet3D', max_num_epochs=1, log_after_iters=2,
                         validate_after_iters=2, max_num_iterations=4, weight_map=False):
        binary_loss = loss in ['BCEWithLogitsLoss', 'DiceLoss', 'GeneralizedDiceLoss']

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        test_config = copy.deepcopy(CONFIG_BASE)
        test_config['model']['name'] = model
        test_config.update({
            # get device to train on
            'device': device,
            'loss': {'name': loss, 'weight': np.random.rand(2).astype(np.float32), 'pos_weight': 3.},
            'eval_metric': {'name': val_metric}
        })
        test_config['model']['final_sigmoid'] = binary_loss

        if weight_map:
            test_config['loaders']['weight_internal_path'] = 'weight_map'

        loss_criterion = get_loss_criterion(test_config)
        eval_criterion = get_evaluation_metric(test_config)
        model = get_model(test_config)
        model = model.to(device)

        if loss in ['BCEWithLogitsLoss']:
            label_dtype = 'float32'
        else:
            label_dtype = 'long'
        test_config['loaders']['train']['transformer']['label'][0]['dtype'] = label_dtype
        test_config['loaders']['val']['transformer']['label'][0]['dtype'] = label_dtype

        train, val = TestUNet3DTrainer._create_random_dataset((3, 128, 128, 128), (3, 64, 64, 64), binary_loss)
        test_config['loaders']['train']['file_paths'] = [train]
        test_config['loaders']['val']['file_paths'] = [val]

        loaders = get_train_loaders(test_config)

        optimizer = _create_optimizer(test_config, model)

        test_config['lr_scheduler']['name'] = 'MultiStepLR'
        lr_scheduler = _create_lr_scheduler(test_config, optimizer)

        logger = get_logger('UNet3DTrainer', logging.DEBUG)

        formatter = DefaultTensorboardFormatter()
        trainer = UNet3DTrainer(model, optimizer, lr_scheduler,
                                loss_criterion, eval_criterion,
                                device, loaders, tmpdir,
                                max_num_epochs=max_num_epochs,
                                log_after_iters=log_after_iters,
                                validate_after_iters=validate_after_iters,
                                max_num_iterations=max_num_iterations,
                                tensorboard_formatter=formatter)
        trainer.fit()
        # test loading the trainer from the checkpoint
        trainer = UNet3DTrainer.from_checkpoint(os.path.join(tmpdir, 'last_checkpoint.pytorch'),
                                                model, optimizer, lr_scheduler,
                                                loss_criterion, eval_criterion,
                                                loaders, tensorboard_formatter=formatter)
        return trainer

    @staticmethod
    def _create_random_dataset(train_shape, val_shape, channel_per_class):
        result = []

        for shape in [train_shape, val_shape]:
            tmp = NamedTemporaryFile(delete=False)

            with h5py.File(tmp.name, 'w') as f:
                l_shape = w_shape = shape
                # make sure that label and weight tensors are 3D
                if len(shape) == 4:
                    l_shape = shape[1:]
                    w_shape = shape[1:]

                if channel_per_class:
                    l_shape = (2,) + l_shape

                f.create_dataset('raw', data=np.random.rand(*shape))
                f.create_dataset('label', data=np.random.randint(0, 2, l_shape))
                f.create_dataset('weight_map', data=np.random.rand(*w_shape))

            result.append(tmp.name)

        return result
