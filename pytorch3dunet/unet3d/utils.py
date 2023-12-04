import importlib
import logging
import os
import shutil
import sys

import h5py
import numpy as np
import torch
from torch import optim


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
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

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def save_network_output(output_path, output, logger=None):
    if logger is not None:
        logger.info(f'Saving network output to: {output_path}...')
    output = output.detach().cpu()[0]
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('predictions', data=output, compression='gzip')


loggers = {}


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_number_of_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def _find_masks(batch, min_size=10):
    """Center the z-slice in the 'middle' of a given instance, given a batch of instances

    Args:
        batch (ndarray): 5d numpy tensor (NCDHW)
    """
    result = []
    for b in batch:
        assert b.shape[0] == 1
        patch = b[0]
        z_sum = patch.sum(axis=(1, 2))
        coords = np.where(z_sum > min_size)[0]
        if len(coords) > 0:
            ind = coords[len(coords) // 2]
            result.append(b[:, ind:ind + 1, ...])
        else:
            ind = b.shape[1] // 2
            result.append(b[:, ind:ind + 1, ...])

    return np.stack(result, axis=0)


def get_tensorboard_formatter(formatter_config):
    if formatter_config is None:
        return DefaultTensorboardFormatter()

    class_name = formatter_config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**formatter_config)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def create_optimizer(optimizer_config, model):
    optim_name = optimizer_config.get('name', 'Adam')
    # common optimizer settings
    learning_rate = optimizer_config.get('learning_rate', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 0)

    # grab optimizer specific settings and init
    # optimizer
    if optim_name == 'Adadelta':
        rho = optimizer_config.get('rho', 0.9)
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=rho,
                                   weight_decay=weight_decay)
    elif optim_name == 'Adagrad':
        lr_decay = optimizer_config.get('lr_decay', 0)
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=lr_decay,
                                  weight_decay=weight_decay)
    elif optim_name == 'AdamW':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas,
                                weight_decay=weight_decay)
    elif optim_name == 'SparseAdam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate, betas=betas)
    elif optim_name == 'Adamax':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, betas=betas,
                                 weight_decay=weight_decay)
    elif optim_name == 'ASGD':
        lambd = optimizer_config.get('lambd', 0.0001)
        alpha = optimizer_config.get('alpha', 0.75)
        t0 = optimizer_config.get('t0', 1e6)
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, lambd=lambd,
                                 alpha=alpha, t0=t0, weight_decay=weight_decay)
    elif optim_name == 'LBFGS':
        max_iter = optimizer_config.get('max_iter', 20)
        max_eval = optimizer_config.get('max_eval', None)
        tolerance_grad = optimizer_config.get('tolerance_grad', 1e-7)
        tolerance_change = optimizer_config.get('tolerance_change', 1e-9)
        history_size = optimizer_config.get('history_size', 100)
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=max_iter,
                                max_eval=max_eval, tolerance_grad=tolerance_grad,
                                tolerance_change=tolerance_change, history_size=history_size)
    elif optim_name == 'NAdam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        momentum_decay = optimizer_config.get('momentum_decay', 4e-3)
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, betas=betas,
                                momentum_decay=momentum_decay,
                                weight_decay=weight_decay)
    elif optim_name == 'RAdam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, betas=betas,
                                weight_decay=weight_decay)
    elif optim_name == 'RMSprop':
        alpha = optimizer_config.get('alpha', 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha,
                                  weight_decay=weight_decay)
    elif optim_name == 'Rprop':
        momentum = optimizer_config.get('momentum', 0)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optim_name == 'SGD':
        momentum = optimizer_config.get('momentum', 0)
        dampening = optimizer_config.get('dampening', 0)
        nesterov = optimizer_config.get('nesterov', False)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                              dampening=dampening, nesterov=nesterov,
                              weight_decay=weight_decay)
    else:  # Adam is default
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas,
                               weight_decay=weight_decay)

    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')
