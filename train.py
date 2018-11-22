import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.hdf5 import AugmentedHDF5Dataset, HDF5Dataset
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import DiceCoefficient
from unet3d.utils import DiceLoss
from unet3d.utils import get_logger
from unet3d.utils import get_number_of_learnable_parameters


def _arg_parser():
    parser = argparse.ArgumentParser(description='UNet3D training')
    parser.add_argument('--checkpoint-dir', required=True,
                        help='checkpoint directory')
    parser.add_argument('--in-channels', required=True, type=int,
                        help='number of input channels')
    parser.add_argument('--out-channels', required=True, type=int,
                        help='number of output channels')
    parser.add_argument('--interpolate',
                        help='use F.interpolate instead of ConvTranspose3d',
                        action='store_true')
    parser.add_argument('--layer-order', type=str,
                        help="Conv layer ordering, e.g. 'crg' -> Conv3D+ReLU+GroupNorm",
                        default='crg')
    parser.add_argument('--loss', type=str, default='bce',
                        help='Which loss function to use. Possible values: [bce, ce, dice]. Where bce - BinaryCrossEntropy (binary classification only), ce - CrossEntropy (multi-class classification), dice - DiceLoss (binary classification only). Default: 20')
    parser.add_argument('--loss-weight', type=float, nargs='+', default=None,
                        help='A manual rescaling weight given to each class in case of CELoss. E.g. --loss-weight 0.1 0.2 0.7')
    parser.add_argument('--epochs', default=500, type=int,
                        help='max number of epochs (default: 500)')
    parser.add_argument('--iters', default=1e5, type=int,
                        help='max number of iterations (default: 1e5)')
    parser.add_argument('--patience', default=20, type=int,
                        help='number of epochs with no loss improvement after which the training will be stopped (default: 20)')
    parser.add_argument('--learning-rate', default=0.0002, type=float,
                        help='initial learning rate (default: 0.0002)')
    parser.add_argument('--weight-decay', default=0.0001, type=float,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--validate-after-iters', default=100, type=int,
                        help='how many iterations between validations (default: 100)')
    parser.add_argument('--log-after-iters', default=100, type=int,
                        help='how many iterations between tensorboard logging (default: 100)')
    parser.add_argument('--resume', type=str,
                        help='path to latest checkpoint (default: none); if provided the training will be resumed from that checkpoint')
    return parser


def _create_model(in_channels, out_channels, layer_order, interpolate=False,
                  final_sigmoid=True):
    return UNet3D(in_channels, out_channels, interpolate, final_sigmoid,
                  conv_layer_order=layer_order)


def _get_loaders(train_path, val_path):
    """
    Returns dictionary containing the  training and validation loaders
    (torch.utils.data.DataLoader) backed by the datasets.hdf5.HDF5Dataset

    :param train_path: path to the H5 file containing the training set
    :param val_path: path to the H5 file containing the validation set
    :param in_channels: number of channels in the raw data
    :param out_channels: number of target channels
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """

    # create H5 backed training dataset with data augmentation
    train_dataset = AugmentedHDF5Dataset(train_path, (32, 64, 64), (16, 32, 32), phase='train')

    # create H5 backed validation dataset
    val_dataset = HDF5Dataset(val_path, (32, 64, 64), (32, 64, 64), phase='val')

    return {
        'train': DataLoader(train_dataset, batch_size=1, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=1, shuffle=True)
    }


def _get_loss_criterion(loss_str, weight=None):
    """
    Returns the loss function together with boolean flag which indicates
    whether to apply an element-wise Sigmoid on the network output
    """
    LOSSES = ['bce', 'dice', 'ce']
    assert loss_str in LOSSES, f'Invalid loss string: {loss_str}'

    if loss_str == 'bce':
        return nn.BCELoss(weight), True
    elif loss_str == 'ce':
        return nn.CrossEntropyLoss(weight), False
    else:
        return DiceLoss(), True


def _create_optimizer(args, model):
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)
    return optimizer


def main():
    parser = _arg_parser()
    logger = get_logger('UNet3DTrainer')
    # Get device to train on
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    logger.info(args)

    # Create loss criterion
    if args.loss_weight is not None:
        loss_weight = torch.tensor(args.loss_weight)
        loss_weight = loss_weight.to(device)
    else:
        loss_weight = None

    loss_criterion, final_sigmoid = _get_loss_criterion(args.loss, loss_weight)

    model = _create_model(args.in_channels, args.out_channels,
                          layer_order=args.layer_order,
                          interpolate=args.interpolate,
                          final_sigmoid=final_sigmoid)

    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(
        f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create error metric
    error_criterion = DiceCoefficient()

    # Get data loaders (for demo purposes the training and validation sets are the same)
    train_path = val_path = 'resources/random.h5'
    loaders = _get_loaders(train_path, val_path)

    # Create the optimizer
    optimizer = _create_optimizer(args, model)

    if args.resume:
        trainer = UNet3DTrainer.from_checkpoint(args.resume, model,
                                                optimizer, loss_criterion,
                                                error_criterion, loaders,
                                                validate_after_iters=args.validate_after_iters,
                                                log_after_iters=args.log_after_iters,
                                                logger=logger)
    else:
        trainer = UNet3DTrainer(model, optimizer, loss_criterion,
                                error_criterion,
                                device, loaders, args.checkpoint_dir,
                                max_num_epochs=args.epochs,
                                max_num_iterations=args.iters,
                                max_patience=args.patience,
                                validate_after_iters=args.validate_after_iters,
                                log_after_iters=args.log_after_iters,
                                logger=logger)

    trainer.fit()


if __name__ == '__main__':
    main()
