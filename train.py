import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import DiceCoefficient
from unet3d.utils import DiceLoss
from unet3d.utils import Random3DDataset
from unet3d.utils import get_logger
from unet3d.utils import get_number_of_learnable_parameters

LOSSES = {
    'bce': (nn.BCELoss(), True),
    'dice': (DiceLoss(), True),
    'ce': (nn.CrossEntropyLoss(), False)
}


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
                        help="Conv layer ordering, e.g. 'brc' -> BatchNorm3d+ReLU+Conv3D",
                        default='brc')
    parser.add_argument('--loss', type=str, default='bce',
                        help='Which loss function to use. Possible values: [bce, ce, dice]. Where bce - BinaryCrossEntropy (binary classification only), ce - CrossEntropy (multi-class classification), dice - DiceLoss (binary classification only)')
    parser.add_argument('--epochs', default=500, type=int,
                        help='max number of epochs (default: 500)')
    parser.add_argument('--iters', default=1e5, type=int,
                        help='max number of iterations (default: 1e5)')
    parser.add_argument('--patience', default=20, type=int,
                        help='number of validation steps with no improvement after which the training will be stopped (default: 20)')
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


def _get_loaders(in_channels, out_channels):
    """Returns dictionary containing the training and validation loaders
    (torch.utils.data.DataLoader) of the form:
    {
        'train': <train_loader>
        'val': <val_loader>
    }"""

    # TODO: replace with your own training and validation loader and don't forget about data augmentation

    # return just a random dataset
    train_dataset = Random3DDataset(4, (32, 64, 64), in_channels, out_channels)
    # same as training data
    val_dataset = train_dataset

    return {
        'train': DataLoader(train_dataset, batch_size=1, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=1, shuffle=True)
    }


def _get_loss_criterion(loss_str):
    """Returns the loss function together with boolean flag which indicates
    whether to use an element-wise Sigmoid on the network output"""
    assert loss_str in LOSSES, f'Invalid loss string: {loss_str}'
    return LOSSES[loss_str]


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
    loss_criterion, final_sigmoid = _get_loss_criterion(args.loss)

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

    # Get data loaders
    loaders = _get_loaders(args.in_channels, args.out_channels)

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
