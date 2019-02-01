import argparse

import yaml


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_train_config():
    parser = argparse.ArgumentParser(description='UNet3D training')
    parser.add_argument('--config', type=str, help='Path to the YAML config file')
    parser.add_argument('--checkpoint-dir', type=str, help='checkpoint directory')
    parser.add_argument('--in-channels', type=int, help='number of input channels')
    parser.add_argument('--out-channels', type=int, help='number of output channels')
    parser.add_argument('--init-channel-number', type=int, default=64,
                        help='Initial number of feature maps in the encoder path which gets '
                             'doubled on every stage (default: 64)')
    parser.add_argument('--interpolate',
                        help='use F.interpolate instead of ConvTranspose3d',
                        action='store_true')
    parser.add_argument('--layer-order', type=str, help="Conv layer ordering, e.g. 'crg' -> Conv3D+ReLU+GroupNorm",
                        default='crg')
    parser.add_argument('--loss', type=str,
                        help='Which loss function to use. Possible values: [bce, ce, wce, dice]. '
                             'Where bce - BinaryCrossEntropyLoss (binary classification only), '
                             'ce - CrossEntropyLoss (multi-class classification), '
                             'wce - WeightedCrossEntropyLoss (multi-class classification), '
                             'dice - GeneralizedDiceLoss (multi-class classification)')
    parser.add_argument('--loss-weight', type=float, nargs='+', default=None,
                        help='A manual rescaling weight given to each class. Can be used with CrossEntropy or BCELoss. '
                             'E.g. --loss-weight 0.3 0.3 0.4')
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
                        help='number of validation rounds with no improvement after which the training '
                             'will be stopped (default: 20)')
    parser.add_argument('--learning-rate', default=0.0002, type=float,
                        help='initial learning rate (default: 0.0002)')
    parser.add_argument('--weight-decay', default=0, type=float,
                        help='weight decay (default: 0)')
    parser.add_argument('--validate-after-iters', default=100, type=int,
                        help='how many iterations between validations (default: 100)')
    parser.add_argument('--log-after-iters', default=100, type=int,
                        help='how many iterations between tensorboard logging (default: 100)')
    parser.add_argument('--resume', type=str,
                        help='path to latest checkpoint (default: none); if provided the training '
                             'will be resumed from that checkpoint')
    parser.add_argument('--train-path', type=str, nargs='+',
                        help='paths to the training datasets, e.g. --train-path <path1> <path2>')
    parser.add_argument('--val-path', type=str, nargs='+',
                        help='paths to the validation datasets, e.g. --val-path <path1> <path2>')
    parser.add_argument('--train-patch', type=int, nargs='+', default=None,
                        help='Patch shape for used for training')
    parser.add_argument('--train-stride', type=int, nargs='+', default=None,
                        help='Patch stride for used for training')
    parser.add_argument('--val-patch', type=int, nargs='+', default=None,
                        help='Patch shape for used for validation')
    parser.add_argument('--val-stride', type=int, nargs='+', default=None,
                        help='Patch stride for used for validation')
    parser.add_argument('--raw-internal-path', type=str, default='raw')
    parser.add_argument('--label-internal-path', type=str, default='label')
    parser.add_argument('--transformer', type=str, default='StandardTransformer', help='data augmentation class')

    args = parser.parse_args()

    if args.config is not None:
        return _load_config_yaml(args.config)

    return args


def parse_test_config():
    parser = argparse.ArgumentParser(description='UNet3D predictions')
    parser.add_argument('--config', type=str, help='Path to the YAML config file')
    parser.add_argument('--model-path', type=str,
                        help='path to the model')
    parser.add_argument('--in-channels', type=int,
                        help='number of input channels')
    parser.add_argument('--out-channels', type=int,
                        help='number of output channels')
    parser.add_argument('--init-channel-number', type=int, default=64,
                        help='Initial number of feature maps in the encoder path; '
                             'the number gets doubled on every stage (default: 64)')
    parser.add_argument('--interpolate',
                        help='use F.interpolate instead of ConvTranspose3d',
                        action='store_true')
    parser.add_argument('--layer-order', type=str,
                        help="Conv layer ordering, e.g. 'crg' -> Conv3D+ReLU+GroupNorm",
                        default='crg')
    parser.add_argument('--final-sigmoid',
                        action='store_true',
                        help='if True apply element-wise nn.Sigmoid after the last layer otherwise apply nn.Softmax')
    parser.add_argument('--test-path', type=str, nargs='+', help='paths to the test dataset')
    parser.add_argument('--raw-internal-path', type=str, default='raw')
    parser.add_argument('--patch', type=int, nargs='+', default=None,
                        help='Patch shape for used for prediction on the test set')
    parser.add_argument('--stride', type=int, nargs='+', default=None,
                        help='Patch stride for used for prediction on the test set')

    args = parser.parse_args()

    if args.config is not None:
        return _load_config_yaml(args.config)

    return args


def _load_config_yaml(config_file):
    config_dict = yaml.load(open(config_file, 'r'))
    return Config(**config_dict)
