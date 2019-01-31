import torch
import torch.optim as optim

from datasets.hdf5 import get_loaders
from unet3d.config import parse_train_config
from unet3d.losses import DiceCoefficient, get_loss_criterion
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger
from unet3d.utils import get_number_of_learnable_parameters


def _create_optimizer(args, model):
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def main():
    logger = get_logger('UNet3DTrainer')
    # Get device to train on
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    config = parse_train_config()

    logger.info(config)

    # Create loss criterion
    if config.loss_weight is not None:
        loss_weight = torch.tensor(config.loss_weight)
        loss_weight = loss_weight.to(device)
    else:
        loss_weight = None

    loss_criterion = get_loss_criterion(config.loss, loss_weight, config.ignore_index)

    model = UNet3D(config.in_channels, config.out_channels,
                   init_channel_number=config.init_channel_number,
                   conv_layer_order=config.layer_order,
                   interpolate=config.interpolate,
                   final_sigmoid=config.final_sigmoid)

    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create accuracy metric
    accuracy_criterion = DiceCoefficient(ignore_index=config.ignore_index)

    # Get data loaders. If 'bce' or 'dice' loss is used, convert labels to float
    train_path, val_path = config.train_path, config.val_path
    if config.loss in ['bce']:
        label_dtype = 'float32'
    else:
        label_dtype = 'long'

    train_patch = tuple(config.train_patch)
    train_stride = tuple(config.train_stride)
    val_patch = tuple(config.val_patch)
    val_stride = tuple(config.val_stride)

    logger.info(f'Train patch/stride: {train_patch}/{train_stride}')
    logger.info(f'Val patch/stride: {val_patch}/{val_stride}')

    pixel_wise_weight = config.loss == 'pce'
    loaders = get_loaders(train_path, val_path, label_dtype=label_dtype,
                          raw_internal_path=config.raw_internal_path, label_internal_path=config.label_internal_path,
                          train_patch=train_patch, train_stride=train_stride,
                          val_patch=val_patch, val_stride=val_stride,
                          transformer=config.transformer, pixel_wise_weight=pixel_wise_weight,
                          curriculum_learning=config.curriculum, ignore_index=config.ignore_index)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    if config.resume:
        trainer = UNet3DTrainer.from_checkpoint(config.resume, model,
                                                optimizer, loss_criterion,
                                                accuracy_criterion, loaders,
                                                logger=logger)
    else:
        trainer = UNet3DTrainer(model, optimizer, loss_criterion,
                                accuracy_criterion,
                                device, loaders, config.checkpoint_dir,
                                max_num_epochs=config.epochs,
                                max_num_iterations=config.iters,
                                max_patience=config.patience,
                                validate_after_iters=config.validate_after_iters,
                                log_after_iters=config.log_after_iters,
                                logger=logger)

    trainer.fit()


if __name__ == '__main__':
    main()
