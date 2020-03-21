import importlib

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import get_logger, get_tensorboard_formatter, get_sample_plotter
from pytorch3dunet.unet3d.utils import get_number_of_learnable_parameters

logger = get_logger('TrainingSetup')


def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    trainer_classname = trainer_config.get('name', 'UNet3DTrainer')
    logger.info(f'NetworkTrainer class: {trainer_classname}')
    # get trainer class
    m = importlib.import_module('pytorch3dunet.unet3d.trainer')
    trainer_class = getattr(m, trainer_classname)

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)
    skip_train_validation = trainer_config.get('skip_train_validation', False)

    # get tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.get('tensorboard_formatter', None))
    # get sample plotter
    sample_plotter = get_sample_plotter(trainer_config.get('sample_plotter', None))

    if resume is not None:
        # continue training from a given checkpoint
        return trainer_class.from_checkpoint(resume, model,
                                             optimizer, lr_scheduler,
                                             loss_criterion, eval_criterion,
                                             loaders,
                                             tensorboard_formatter=tensorboard_formatter,
                                             sample_plotter=sample_plotter,
                                             skip_train_validation=skip_train_validation)
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return trainer_class.from_pretrained(pre_trained, model,
                                             optimizer, lr_scheduler,
                                             loss_criterion, eval_criterion,
                                             device=config['device'], loaders=loaders,
                                             max_num_epochs=trainer_config['epochs'],
                                             max_num_iterations=trainer_config['iters'],
                                             validate_after_iters=trainer_config['validate_after_iters'],
                                             log_after_iters=trainer_config['log_after_iters'],
                                             eval_score_higher_is_better=trainer_config[
                                                 'eval_score_higher_is_better'],
                                             tensorboard_formatter=tensorboard_formatter,
                                             sample_plotter=sample_plotter,
                                             skip_train_validation=skip_train_validation)
    else:
        # start training from scratch
        return trainer_class(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                             config['device'], loaders, trainer_config['checkpoint_dir'],
                             max_num_epochs=trainer_config['epochs'],
                             max_num_iterations=trainer_config['iters'],
                             validate_after_iters=trainer_config['validate_after_iters'],
                             log_after_iters=trainer_config['log_after_iters'],
                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                             tensorboard_formatter=tensorboard_formatter,
                             sample_plotter=sample_plotter,
                             skip_train_validation=skip_train_validation)


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is not None:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)
    return None


def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    model = get_model(config)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
