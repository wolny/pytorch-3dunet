import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from augment.transforms import AnisotropicRotationTransformer, RandomLabelToBoundaryTransformer, \
    IsotropicRotationTransformer, \
    StandardTransformer, BaseTransformer, LabelToBoundaryTransformer
from datasets.hdf5 import HDF5Dataset, CurriculumLearningSliceBuilder, SliceBuilder, get_loaders
from unet3d.losses import DiceCoefficient, get_loss_criterion
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger
from unet3d.utils import get_number_of_learnable_parameters

import yaml


def _load_config():
    parser = argparse.ArgumentParser(description='UNet3D training')
    parser.add_argument('--config', required=True, type=str, help='Config file path')
    config = yaml.load(open(parser.parse_args().config))
    return config


def _get_loaders(train_path, val_path, raw_internal_path, label_internal_path, label_dtype, train_patch, train_stride,
                 val_patch, val_stride, transformer, pixel_wise_weight=False, curriculum_learning=False,
                 ignore_index=None):
    """
    Returns dictionary containing the  training and validation loaders
    (torch.utils.data.DataLoader) backed by the datasets.hdf5.HDF5Dataset

    :param train_path: path to the H5 file containing the training set
    :param val_path: path to the H5 file containing the validation set
    :param raw_internal_path:
    :param label_internal_path:
    :param label_dtype: target type of the label dataset
    :param train_patch:
    :param train_stride:
    :param val_path:
    :param val_stride:
    :param transformer:
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    transformers = {
        'LabelToBoundaryTransformer': LabelToBoundaryTransformer,
        'RandomLabelToBoundaryTransformer': RandomLabelToBoundaryTransformer,
        'AnisotropicRotationTransformer': AnisotropicRotationTransformer,
        'IsotropicRotationTransformer': IsotropicRotationTransformer,
        'StandardTransformer': StandardTransformer,
        'BaseTransformer': BaseTransformer
    }

    assert transformer in transformers

    if curriculum_learning:
        slice_builder_cls = CurriculumLearningSliceBuilder
    else:
        slice_builder_cls = SliceBuilder

    # create H5 backed training and validation dataset with data augmentation
    train_dataset = HDF5Dataset(train_path, train_patch, train_stride,
                                phase='train',
                                label_dtype=label_dtype,
                                raw_internal_path=raw_internal_path,
                                label_internal_path=label_internal_path,
                                transformer=transformers[transformer],
                                weighted=pixel_wise_weight,
                                ignore_index=ignore_index,
                                slice_builder_cls=slice_builder_cls)

    val_dataset = HDF5Dataset(val_path, val_patch, val_stride,
                              phase='val',
                              label_dtype=label_dtype,
                              raw_internal_path=raw_internal_path,
                              label_internal_path=label_internal_path,
                              transformer=transformers[transformer],
                              weighted=pixel_wise_weight,
                              ignore_index=ignore_index)

    # shuffle only if curriculum_learning scheme is not used
    return {
        'train': DataLoader(train_dataset, batch_size=1, shuffle=not curriculum_learning),
        'val': DataLoader(val_dataset, batch_size=1, shuffle=not curriculum_learning)
    }


def _create_optimizer(config, model):
    learning_rate = config['learning-rate']
    weight_decay = config['weight-decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def main():
    config = _load_config()
    logger = get_logger('UNet3DTrainer')
    # Get device to train on
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    logger.info(config)

    # Create loss criterion
    if config['loss-weight'] is not None:
        loss_weight = torch.tensor(config['loss-weight'])
        loss_weight = loss_weight.to(device)
    else:
        loss_weight = None

    loss_criterion = get_loss_criterion(config['loss'],
                                        config['final-sigmoid'],
                                        loss_weight,
                                        config['ignore-index'])

    model = UNet3D(config['in-channels'],
                   config['out-channels'],
                   init_channel_number=config['init-channel-number'],
                   conv_layer_order=config['layer-order'],
                   interpolate=config['interpolate'],
                   final_sigmoid=config['final-sigmoid'])

    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create accuracy metric
    accuracy_criterion = DiceCoefficient(ignore_index=config['ignore-index'])

    # Get data loaders. If 'bce' or 'dice' loss is used, convert labels to float
    train_path, val_path = config['train-path'], config['val-path']
    if config['loss'] in ['bce', 'dice']:
        label_dtype = 'float32'
    else:
        label_dtype = 'long'

    train_patch = tuple(config['train-patch'])
    train_stride = tuple(config['train-stride'])
    val_patch = tuple(config['val-patch'])
    val_stride = tuple(config['val-stride'])

    logger.info(f'Train patch/stride: {train_patch}/{train_stride}')
    logger.info(f'Val patch/stride: {val_patch}/{val_stride}')

    pixel_wise_weight = config['loss'] == 'pce'
    loaders = get_loaders(train_path, val_path, label_dtype=label_dtype,
                           raw_internal_path=config['raw-internal-path'],
                           label_internal_path=config['label-internal-path'],
                           train_patch=train_patch, train_stride=train_stride,
                           val_patch=val_patch, val_stride=val_stride,
                           transformer=config['transformer'], pixel_wise_weight=pixel_wise_weight,
                           curriculum_learning=config['curriculum'], ignore_index=config['ignore-index'])

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    if config['resume']:
        trainer = UNet3DTrainer.from_checkpoint(config['resume'], model,
                                                optimizer, loss_criterion,
                                                accuracy_criterion, loaders,
                                                logger=logger)
    else:
        trainer = UNet3DTrainer(model, optimizer, loss_criterion,
                                accuracy_criterion,
                                device, loaders, config['checkpoint-dir'],
                                max_num_epochs=config['epochs'],
                                max_num_iterations=int(config['iters']),
                                max_patience=config['patience'],
                                validate_after_iters=config['validate-after-iters'],
                                log_after_iters=config['log-after-iters'],
                                logger=logger)

    trainer.fit()


if __name__ == '__main__':
    main()
