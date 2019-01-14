import argparse

import torch
import torch.optim as optim

from datasets.hdf5 import get_loaders
from unet3d.losses import DiceCoefficient, get_loss_criterion
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger
from unet3d.utils import get_number_of_learnable_parameters


def _arg_parser():
    parser = argparse.ArgumentParser(description='UNet3D training')
    parser.add_argument('--checkpoint-dir', help='checkpoint directory')
    parser.add_argument('--in-channels', required=True, type=int,
                        help='number of input channels')
    parser.add_argument('--out-channels', required=True, type=int,
                        help='number of output channels')
    parser.add_argument('--init-channel-number', type=int, default=64,
                        help='Initial number of feature maps in the encoder path which gets doubled on every stage (default: 64)')
    parser.add_argument('--interpolate',
                        help='use F.interpolate instead of ConvTranspose3d',
                        action='store_true')
    parser.add_argument('--layer-order', type=str,
                        help="Conv layer ordering, e.g. 'crg' -> Conv3D+ReLU+GroupNorm",
                        default='crg')
    parser.add_argument('--loss', type=str, required=True,
                        help='Which loss function to use. Possible values: [bce, ce, wce, dice]. Where bce - BinaryCrossEntropyLoss (binary classification only), ce - CrossEntropyLoss (multi-class classification), wce - WeightedCrossEntropyLoss (multi-class classification), dice - GeneralizedDiceLoss (multi-class classification)')
    parser.add_argument('--loss-weight', type=float, nargs='+', default=None,
                        help='A manual rescaling weight given to each class. Can be used with CrossEntropy or BCELoss. E.g. --loss-weight 0.3 0.3 0.4')
    parser.add_argument('--ignore-index', type=int, default=None,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')
    parser.add_argument('--curriculum',
                        help='use simple Curriculum Learning scheme if ignore_index is present',
                        action='store_true')
    parser.add_argument('--final-sigmoid',
                        action='store_true',
                        help='if True apply element-wise nn.Sigmoid after the last layer otherwise apply nn.Softmax')
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
    parser.add_argument('--train-path', type=str, nargs='+', required=True, help='paths to the training datasets, e.g. --train-path <path1> <path2>')
    parser.add_argument('--val-path', type=str, nargs='+', required=True, help='paths to the validation datasets, e.g. --val-path <path1> <path2>')
    parser.add_argument('--train-patch', required=True, type=int, nargs='+', default=None,
                        help='Patch shape for used for training')
    parser.add_argument('--train-stride', required=True, type=int, nargs='+', default=None,
                        help='Patch stride for used for training')
    parser.add_argument('--val-patch', required=True, type=int, nargs='+', default=None,
                        help='Patch shape for used for validation')
    parser.add_argument('--val-stride', required=True, type=int, nargs='+', default=None,
                        help='Patch stride for used for validation')
    parser.add_argument('--raw-internal-path', type=str, default='raw')
    parser.add_argument('--label-internal-path', type=str, default='label')
    parser.add_argument('--transformer', type=str, default='StandardTransformer', help='data augmentation class')

    return parser


def _create_optimizer(args, model):
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

    loss_criterion = get_loss_criterion(args.loss, args.final_sigmoid, loss_weight, args.ignore_index)

    model = UNet3D(args.in_channels, args.out_channels,
                   init_channel_number=args.init_channel_number,
                   conv_layer_order=args.layer_order,
                   interpolate=args.interpolate,
                   final_sigmoid=args.final_sigmoid)

    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create accuracy metric
    accuracy_criterion = DiceCoefficient(ignore_index=args.ignore_index)

    # Get data loaders. If 'bce' or 'dice' loss is used, convert labels to float
    train_path, val_path = args.train_path, args.val_path
    if args.loss in ['bce', 'dice']:
        label_dtype = 'float32'
    else:
        label_dtype = 'long'

    train_patch = tuple(args.train_patch)
    train_stride = tuple(args.train_stride)
    val_patch = tuple(args.val_patch)
    val_stride = tuple(args.val_stride)

    logger.info(f'Train patch/stride: {train_patch}/{train_stride}')
    logger.info(f'Val patch/stride: {val_patch}/{val_stride}')

    pixel_wise_weight = args.loss == 'pce'
    loaders = get_loaders(train_path, val_path, label_dtype=label_dtype,
                          raw_internal_path=args.raw_internal_path, label_internal_path=args.label_internal_path,
                          train_patch=train_patch, train_stride=train_stride,
                          val_patch=val_patch, val_stride=val_stride,
                          transformer=args.transformer, pixel_wise_weight=pixel_wise_weight,
                          curriculum_learning=args.curriculum, ignore_index=args.ignore_index)

    # Create the optimizer
    optimizer = _create_optimizer(args, model)

    if args.resume:
        trainer = UNet3DTrainer.from_checkpoint(args.resume, model,
                                                optimizer, loss_criterion,
                                                accuracy_criterion, loaders,
                                                logger=logger)
    else:
        trainer = UNet3DTrainer(model, optimizer, loss_criterion,
                                accuracy_criterion,
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
