import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.config import TorchDevice
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model, is_model_2d
from pytorch3dunet.unet3d.utils import get_logger, create_optimizer, \
    create_lr_scheduler, get_number_of_learnable_parameters, TensorboardFormatter
from . import utils

logger = get_logger('UNetTrainer')


def create_trainer(config: dict) -> 'UNetTrainer':
    # Create the model
    model = get_model(config['model'])

    device = config.get("device", None)
    assert device, "Device not specified in the config file and could not be inferred automatically"
    logger.info(f'Using device: {device}')

    # use DataParallel if more than 1 GPU available
    if device == TorchDevice.CUDA and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
    model.to(device)

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = create_optimizer(config['optimizer'], model)

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

    trainer_config = config['trainer']
    # Create tensorboard formatter
    tensorboard_formatter_config = trainer_config.pop('tensorboard_formatter', {})
    tensorboard_formatter = TensorboardFormatter(**tensorboard_formatter_config)
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return UNetTrainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_criterion=loss_criterion,
                       eval_criterion=eval_criterion, loaders=loaders, tensorboard_formatter=tensorboard_formatter,
                       resume=resume, pre_trained=pre_trained, device=device, **trainer_config)


def _split_and_move_to_gpu(t, device: TorchDevice):
    def _move_to_gpu(input, device):
        if isinstance(input, (tuple, list)):
            return tuple([_move_to_gpu(x, device) for x in input])
        else:
            if device == TorchDevice.CUDA:
                input = input.cuda(non_blocking=True)
            else:
                input = input.to(device)

            return input

    input, target = _move_to_gpu(t, device)
    return input, target


class UNetTrainer:
    """UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (str): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used when
            evaluation is expensive)
        resume (str): path to the checkpoint to be resumed
        pre_trained (str): path to the pre-trained model
        max_val_images (int): maximum number of images to log during validation
        device (TorchDevice): device to use for training (CPU, CUDA, MPS)
    """

    def __init__(
            self,
            model,
            optimizer,
            lr_scheduler,
            loss_criterion,
            eval_criterion,
            loaders,
            checkpoint_dir,
            max_num_epochs,
            max_num_iterations,
            validate_after_iters=200,
            log_after_iters=100,
            validate_iters=None,
            num_iterations=1,
            num_epoch=0,
            eval_score_higher_is_better=True,
            tensorboard_formatter=None,
            skip_train_validation=False,
            resume=None,
            pre_trained=None,
            max_val_images=100,
            device: Optional[TorchDevice] = None
    ):

        self.max_val_images = max_val_images
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        assert device, "Device must be specified"
        self.device = device

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')
        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(
            log_dir=os.path.join(
                checkpoint_dir, 'logs',
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
        )

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = utils.load_checkpoint(resume, self.model, self.optimizer)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            utils.load_checkpoint(pre_trained, self.model, None)
            if not self.checkpoint_dir:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        for _ in range(self.num_epochs, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

            input, target = _split_and_move_to_gpu(t, self.device)

            output, loss = self._forward_pass(input, target)

            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score = self.validate()
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                elif self.scheduler is not None:
                    self.scheduler.step()

                # log current learning rate in tensorboard
                self._log_lr()
                # remember the best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                # compute eval criterion
                if not self.skip_train_validation:
                    eval_score = self.eval_criterion(output, target)
                    train_eval_scores.update(eval_score.item(), self._batch_size(input))

                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_images(
                    input.detach().cpu().numpy(),
                    target.detach().cpu().numpy(),
                    output.detach().cpu().numpy(),
                    'train_'
                )

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            # select indices of validation samples to log
            rs = np.random.RandomState(42)
            if len(self.loaders['val']) <= self.max_val_images:
                indices = list(range(len(self.loaders['val'])))
            else:
                indices = rs.choice(len(self.loaders['val']), size=self.max_val_images, replace=False)

            images_for_logging = []
            for i, t in enumerate(tqdm(self.loaders['val'])):
                input, target = _split_and_move_to_gpu(t, self.device)

                output, loss = self._forward_pass(input, target)
                val_losses.update(loss.item(), self._batch_size(input))
                eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

                # save val images for logging
                if i in indices:
                    imgs = (input.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy())
                    images_for_logging.append(imgs + (i,))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            # log images in a separate thread
            with ThreadPoolExecutor() as executor:
                for input, target, output, i in images_for_logging:
                    executor.submit(self._log_images, input, target, output, f'val_{i}_')

            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            self._log_stats('val', val_losses.avg, val_scores.avg)
            return val_scores.avg

    def _forward_pass(self, inp: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if is_model_2d(self.model):
            # remove the singleton z-dimension from the input
            inp = torch.squeeze(inp, dim=-3)
            # forward pass
            output, logits = self.model(inp, return_logits=True)
            # add the singleton z-dimension to the output
            output = torch.unsqueeze(output, dim=-3)
            logits = torch.unsqueeze(logits, dim=-3)
        else:
            # forward pass
            output, logits = self.model(inp, return_logits=True)

        # always compute the loss using logits
        loss = self.loss_criterion(logits, target)

        # return probabilities and loss
        return output, loss

    def _is_best_eval_score(self, eval_score: float) -> bool:
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best: bool):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        utils.save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase: str, loss_avg: float, eval_score_avg: float):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(
            self,
            input: np.ndarray,
            target: np.ndarray,
            prediction: np.ndarray,
            prefix: str):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, (list, tuple)):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b
            else:
                img_sources[name] = batch

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input: torch.Tensor) -> int:
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
