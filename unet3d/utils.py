import logging
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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


class Random3DDataset(Dataset):
    """Generates random 3D dataset for testing and demonstration purposes.
    Args:
        N (int): batch size
        size (tuple): dimensionality of each batch (DxHxW)
        in_channels (int): number of input channels
        out_channels (int): number of output channels (labeled masks)
    """

    def __init__(self, N, size, in_channels, out_channels):
        assert len(size) == 3
        raw_dims = (N, in_channels) + size
        labels_dims = (N, out_channels) + size
        self.raw = torch.randn(raw_dims)
        self.labels = torch.empty(labels_dims, dtype=torch.float).random_(2)

    def __len__(self):
        return self.raw.size(0)

    def __getitem__(self, idx):
        """Returns tuple (raw, labels) for a given batch 'idx'"""
        return self.raw[idx], self.labels[idx]


class RandomSliced3DDataset(Dataset):
    """Generates random 3D dataset for testing purposes.
    Args:
        raw_shape (tuple): shape of the randomly generated dataset (CxDxHxW)
        patch_shape (tuple): shape of the patch DxHxW
        stride_shape (tuple): shape of the stride DxHxW
    """

    def __init__(self, raw_shape, patch_shape, stride_shape):
        assert len(raw_shape) == 4
        assert len(patch_shape) == 3
        assert len(stride_shape) == 3

        self.raw = np.random.randn(*raw_shape).astype('float32')

        self.raw_slices = RandomSliced3DDataset.build_slices(raw_shape,
                                                             patch_shape,
                                                             stride_shape)

    def __len__(self):
        return len(self.raw_slices)

    def __getitem__(self, idx):
        """Returns the tuple (raw, slice)"""
        if idx not in self.raw_slices:
            raise StopIteration()

        index_spec = self.raw_slices[idx]
        return torch.from_numpy(self.raw[index_spec]), index_spec[1:]

    @staticmethod
    def build_slices(raw_shape, patch_shape, stride_shape):
        """Iterates over the n-dimensional array of shape 'raw_shape' patch-by-patch
        using the patch shape of 'patch_shape' and stride of 'stride_shape' and
        builds the mapping from index to slice position.

        Args:
            raw_shape (tuple): shape of the n-dim array
            patch_shape (tuple): patch shape
            stride_shape (tuple): stride shape

        Returns:
            slice mapping (int -> (slice, slice, slice, slice))
        """
        slices = {}
        in_channels, i_z, i_y, i_x = raw_shape
        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        idx = 0
        z_steps = RandomSliced3DDataset._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = RandomSliced3DDataset._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = RandomSliced3DDataset._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slices[idx] = (
                        slice(0, in_channels),
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    idx += 1
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        for j in range(0, i - k + 1, s):
            yield j
        if not (i - k + 1) % k == 0:
            yield i - k


class DiceCoefficient(nn.Module):
    """Compute Dice Coefficient averaging across batch axis
    """

    def __init__(self, epsilon=1e-5):
        super(DiceCoefficient, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        assert input.size() == target.size()

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
