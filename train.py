import torch.optim as optim

from datasets.hdf5 import get_train_loaders
from unet3d.config import load_config
from unet3d.losses import get_loss_criterion
from unet3d.metrics import get_evaluation_metric
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger
from unet3d.utils import get_number_of_learnable_parameters


def _create_optimizer(config, model):
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def main():
    logger = get_logger('UNet3DTrainer')

    config = load_config()

    logger.info(config)

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)

    # Create the model
    model = UNet3D(config['in_channels'], config['out_channels'],
                   final_sigmoid=config['final_sigmoid'],
                   init_channel_number=config['init_channel_number'],
                   conv_layer_order=config['layer_order'],
                   interpolate=config['interpolate'])

    model = model.to(config['device'])

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    if config['resume'] is not None:
        trainer = UNet3DTrainer.from_checkpoint(config['resume'], model,
                                                optimizer, loss_criterion,
                                                eval_criterion, loaders,
                                                logger=logger)
    else:
        trainer = UNet3DTrainer(model, optimizer, loss_criterion, eval_criterion,
                                config['device'], loaders, config['checkpoint_dir'],
                                max_num_epochs=config['epochs'],
                                max_num_iterations=config['iters'],
                                max_patience=config['patience'],
                                validate_after_iters=config['validate_after_iters'],
                                log_after_iters=config['log_after_iters'],
                                logger=logger)

    trainer.fit()


if __name__ == '__main__':
    main()
