import logging
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best validation error so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class ComposedLoss(nn.Module):
    """Helper class for composed loss functions.

    Args:
        input_func (nn.Module): element-wise function applied on the input before
            passing it to the output
        loss (nn.Module): loss function to be applied on the transformed input and target

    Example:
        ```
        loss = ComposedLoss(nn.Sigmoid(), nn.BCELoss())
        output = loss(input, target)
        ```
        would be equivalent to:
        ```
        loss = nn.BCELoss()
        output = loss(F.sigmoid(input), target)
        ```
    """

    def __init__(self, input_func, loss):
        super(ComposedLoss, self).__init__()
        self.input_func = input_func
        self.loss = loss

    def forward(self, input, target):
        return self.loss(self.input_func(input), target)


class DiceCoefficient(nn.Module):
    """Compute Dice Coefficient averaging across batch axis
    """

    def __init__(self, epsilon=1e-5, expand_dims=True):
        super(DiceCoefficient, self).__init__()
        self.epsilon = epsilon
        self.expand_dims = expand_dims

    def forward(self, input, target):
        if not self.expand_dims:
            assert input.size() == target.size()
        else:
            # input NxCxDxHxW, target NxDxHxW
            if input.size() != target.size():
                t = torch.zeros_like(input)
                # for each batch instance
                for i in range(input.size()[0]):
                    # iterate over channel axis and create corresponding binary mask in the target
                    for c in range(input.size()[1]):
                        mask = t[i]
                        mask[mask == c] = 1
                target = t

        inter_card = (input * target).sum()
        sum_of_cards = input.sum() + target.sum()
        return (2. * inter_card + self.epsilon) / (sum_of_cards + self.epsilon)


class DiceLoss(DiceCoefficient):
    """Compute Dice Loss averaging across batch axis.
    Just the negation of Dice Coefficient.
    """

    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__(epsilon)

    def forward(self, input, target):
        coeff = super(DiceLoss, self).forward(input, target)
        return -1.0 * coeff


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)
