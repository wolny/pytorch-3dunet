import importlib
import logging
import os
import shutil
import sys
from typing import Any

import numpy as np
import torch
from skimage.color import label2rgb
from torch import nn, optim
from torch.optim import Optimizer


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state: contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best: if True state contains the best model seen so far
        checkpoint_dir: directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, "last_checkpoint.pytorch")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, "best_checkpoint.pytorch")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optimizer = None,
    model_key: str = "model_state_dict",
    optimizer_key: str = "optimizer_state_dict",
) -> dict:
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path: path to the checkpoint to be loaded
        model: model into which the parameters are to be copied
        optimizer: optimizer instance into which the parameters are to be copied
        model_key: key corresponding to the model state_dict in the checkpoint
        optimizer_key: key corresponding to the optimizer state_dict in the checkpoint

    Returns:
        state dict stored in the checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise OSError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


loggers = {}


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Initializes and returns a logger with the given name."""

    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_number_of_learnable_parameters(model: nn.Module) -> int:
    """Returns the number of learnable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RunningAverage:
    """Computes and stores the average"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def number_of_features_per_level(init_channel_number: int, num_levels: int) -> list:
    """Computes the number of features at each level of the UNet"""
    return [init_channel_number * 2**k for k in range(num_levels)]


class TensorboardFormatter:
    """
    Tensorboard formatter converts a given batch of images (input/prediction/target)
    to a series of images that can be displayed in tensorboard.

    Args:
        skip_last_target (bool): if True, the last channel of the target image is skipped. This is useful for boundary
            based segmentation where the first channel is the boundary map and the second channel is the actual
            segmentation not used for training.
        log_channelwise (bool): if True, logs each channel of a multi-channel prediction, if False, takes the argmax
            over the channel dimension and logs a single label image.
    """

    def __init__(self, skip_last_target=False, log_channelwise=False):
        self.skip_last_target = skip_last_target
        self.log_channelwise = log_channelwise

    def __call__(self, name: str, batch: np.ndarray) -> list:
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name: one of 'inputs'/'targets'/'predictions'
             batch: 4D or 5D numpy array

        Returns:
            list[(str, np.ndarray)]: list of tuples of the form (tag, img)
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, "Only 2D (HW) and 3D (CHW) images are accepted for display"

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, "Only (1, H, W) or (3, H, W) images are supported"

            return tag, img

        tagged_images = self._process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def _process_batch(self, name: str, batch: np.ndarray) -> list:
        if name == "targets" and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = "{}/batch_{}/slice_{}"

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                if self.log_channelwise and name == "predictions":
                    tag_template = "{}/batch_{}/channel_{}/slice_{}"
                    for channel_idx in range(batch.shape[1]):
                        tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                        img = batch[batch_idx, channel_idx, slice_idx, ...]
                        tagged_images.append((tag, self._normalize_img(img)))
                else:
                    tag = tag_template.format(name, batch_idx, slice_idx)
                    if name in ["predictions", "targets"]:
                        # for single channel predictions, just log the image
                        if batch.shape[1] == 1:
                            img = batch[batch_idx, :, slice_idx, ...]
                            tagged_images.append((tag, self._normalize_img(img)))
                        else:
                            # predictions are probabilities so convert to label image
                            img = batch[batch_idx].argmax(axis=0)
                            # take the middle slice
                            img = img[slice_idx, ...]
                            # convert to label image
                            img = label2rgb(img)
                            img = img.transpose(2, 0, 1)
                            tagged_images.append((tag, img))
                    else:
                        # handle input images
                        if batch.shape[1] in [1, 3]:
                            # if single channel or RGB image, log directly
                            img = batch[batch_idx, :, slice_idx, ...]
                            tagged_images.append((tag, self._normalize_img(img)))
                        else:
                            # log channelwise
                            tag_template = "{}/batch_{}/channel_{}/slice_{}"
                            for channel_idx in range(batch.shape[1]):
                                tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                                img = batch[batch_idx, channel_idx, slice_idx, ...]
                                tagged_images.append((tag, self._normalize_img(img)))

        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                # this is target segmentation so convert to label image
                lbl = label2rgb(img)
                lbl = lbl.transpose(2, 0, 1)
                tagged_images.append((tag, lbl))

        return tagged_images

    @staticmethod
    def _normalize_img(img: np.ndarray) -> np.ndarray:
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


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


def create_optimizer(optimizer_config: dict, model: nn.Module) -> Optimizer:
    optim_name = optimizer_config.get("name", "Adam")
    # common optimizer settings
    learning_rate = optimizer_config.get("learning_rate", 1e-3)
    weight_decay = optimizer_config.get("weight_decay", 0)

    # grab optimizer specific settings and init
    # optimizer
    if optim_name == "Adadelta":
        rho = optimizer_config.get("rho", 0.9)
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=rho, weight_decay=weight_decay)
    elif optim_name == "Adagrad":
        lr_decay = optimizer_config.get("lr_decay", 0)
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=lr_decay, weight_decay=weight_decay)
    elif optim_name == "AdamW":
        betas = tuple(optimizer_config.get("betas", (0.9, 0.999)))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optim_name == "SparseAdam":
        betas = tuple(optimizer_config.get("betas", (0.9, 0.999)))
        optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate, betas=betas)
    elif optim_name == "Adamax":
        betas = tuple(optimizer_config.get("betas", (0.9, 0.999)))
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optim_name == "LBFGS":
        max_iter = optimizer_config.get("max_iter", 20)
        max_eval = optimizer_config.get("max_eval", None)
        tolerance_grad = optimizer_config.get("tolerance_grad", 1e-7)
        tolerance_change = optimizer_config.get("tolerance_change", 1e-9)
        history_size = optimizer_config.get("history_size", 100)
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
        )
    elif optim_name == "NAdam":
        betas = tuple(optimizer_config.get("betas", (0.9, 0.999)))
        momentum_decay = optimizer_config.get("momentum_decay", 4e-3)
        optimizer = optim.NAdam(
            model.parameters(), lr=learning_rate, betas=betas, momentum_decay=momentum_decay, weight_decay=weight_decay
        )
    elif optim_name == "RAdam":
        betas = tuple(optimizer_config.get("betas", (0.9, 0.999)))
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optim_name == "RMSprop":
        alpha = optimizer_config.get("alpha", 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay)
    elif optim_name == "Rprop":
        etas = optimizer_config.get("etas", (0.5, 1.2))
        step_sizes = optimizer_config.get("step_sizes", (1e-6, 50))
        optimizer = optim.Rprop(model.parameters(), lr=learning_rate, etas=etas, step_sizes=step_sizes)
    elif optim_name == "SGD":
        momentum = optimizer_config.get("momentum", 0)
        dampening = optimizer_config.get("dampening", 0)
        nesterov = optimizer_config.get("nesterov", False)
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    else:  # Adam is default
        betas = tuple(optimizer_config.get("betas", (0.9, 0.999)))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    return optimizer


def create_lr_scheduler(lr_config: dict, optimizer: Optimizer) -> Any | None:
    """Creates a learning rate scheduler"""
    if lr_config is None:
        return None
    class_name = lr_config.pop("name")
    m = importlib.import_module("torch.optim.lr_scheduler")
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config["optimizer"] = optimizer
    return clazz(**lr_config)


def get_class(class_name: str, modules: list[str]) -> type:
    """Helper function which searches for a class in the given list of modules and returns it."""
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported dataset class: {class_name}")
