import logging
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

from . import utils


class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        loss_criterion (callable): loss function
        accuracy_criterion (callable): used to compute training/validation accuracy (such as Dice or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        max_patience (int): number of validation runs with no improvement
            after which the training will be stopped
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        best_val_accuracy (float): best validation accuracy so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, loss_criterion, accuracy_criterion,
                 device, loaders, checkpoint_dir,
                 max_num_epochs=200, max_num_iterations=1e5, max_patience=20,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, best_val_accuracy=float('-inf'),
                 num_iterations=0, num_epoch=0, logger=None):
        if logger is None:
            self.logger = utils.get_logger('UNet3DTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        self.logger.info(f"Sending the model to '{device}'")
        self.model = model.to(device)
        self.logger.debug(model)

        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.accuracy_criterion = accuracy_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.best_val_accuracy = best_val_accuracy
        self.writer = SummaryWriter(
            log_dir=os.path.join(checkpoint_dir, 'logs'))

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        # used for early stopping
        self.max_patience = max_patience
        self.patience = max_patience

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, loss_criterion, accuracy_criterion, loaders,
                        logger=None):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val accuracy: {state['best_val_accuracy']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, loss_criterion, accuracy_criterion, torch.device(state['device']), loaders,
                   checkpoint_dir,
                   best_val_accuracy=state['best_val_accuracy'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   max_patience=state['max_patience'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   logger=logger)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch += 1

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_accuracy = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        for i, t in enumerate(train_loader):
            self.logger.info(
                f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            if len(t) == 2:
                input, target = t
                input, target = input.to(self.device), target.to(self.device)
                weight = None
            else:
                input, target, weight = t
                input, target, weight = input.to(self.device), target.to(self.device), weight.to(self.device)

            if hasattr(self.loss_criterion, 'ignore_index') and self.loss_criterion.ignore_index is not None:
                unique_labels = torch.unique(target)
                if len(unique_labels) == 1 and unique_labels.item() == self.loss_criterion.ignore_index:
                    self.logger.info(f'Skipping training batch {i} (contains only ignore_index)...')
                    continue

            output, loss, accuracy = self._forward_pass(input, target, weight)

            train_losses.update(loss.item(), input.size(0))
            train_accuracy.update(accuracy.item(), input.size(0))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.num_iterations += 1

            if self.num_iterations % self.log_after_iters == 0:
                # log stats, params and images
                self.logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Accuracy: {train_accuracy.avg}')
                self._log_stats('train', train_losses.avg, train_accuracy.avg)
                self._log_params()
                # normalize output (during training the network outputs logits only)
                output = self.model.final_activation(output)
                self._log_images(input, target, output)

            if self.num_iterations % self.validate_after_iters == 0:
                # evaluate on validation set
                val_accuracy = self.validate(self.loaders['val'])

                # remember best validation metric
                is_best = self._is_best_val_accuracy(val_accuracy)

                # save checkpoint
                self._save_checkpoint(is_best)

                if self._check_early_stopping(is_best):
                    self.logger.info(
                        f'Validation accuracy did not improve for the last {self.max_patience} validation runs. Early stopping...')
                    return True

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

        return False

    def validate(self, val_loader):
        self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_accuracy = utils.RunningAverage()

        try:
            with torch.no_grad():
                for i, t in enumerate(val_loader):
                    self.logger.info(f'Validation iteration {i}')

                    if len(t) == 2:
                        input, target = t
                        input, target = input.to(self.device), target.to(self.device)
                        weight = None
                    else:
                        input, target, weight = t
                        input, target, weight = input.to(self.device), target.to(self.device), weight.to(self.device)

                    if hasattr(self.loss_criterion, 'ignore_index') and self.loss_criterion.ignore_index is not None:
                        unique_labels = torch.unique(target)
                        if len(unique_labels) == 1 and unique_labels.item() == self.loss_criterion.ignore_index:
                            self.logger.info(f'Skipping validation batch {i} (contains only ignore_index)...')
                            continue

                    output, loss, accuracy = self._forward_pass(input, target, weight)

                    val_losses.update(loss.item(), input.size(0))
                    val_accuracy.update(accuracy.item(), input.size(0))

                    if self.validate_iters is not None and self.validate_iters <= i:
                        # stop validation
                        break

                self._log_stats('val', val_losses.avg, val_accuracy.avg)
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Accuracy: {val_accuracy.avg}')
                return val_accuracy.avg
        finally:
            self.model.train()

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        # normalize logits and compute the accuracy criterion
        accuracy = self.accuracy_criterion(self.model.final_activation(output), target)

        return output, loss, accuracy

    def _check_early_stopping(self, best_model_found):
        """
        Check patience and adjust the learning rate if necessary.
        :param best_model_found: is current model the best one according to validation criterion
        :return: True if the training should be terminated, False otherwise
        """
        if best_model_found:
            self.patience = self.max_patience
        else:
            self.patience -= 1
            if self.patience <= 0:
                # early stop the training
                return True
            # adjust learning rate when reaching half of the max_patience
            if self.patience == self.max_patience // 2:
                self._adjust_learning_rate()
                self.patience = self.max_patience
        return False

    def _adjust_learning_rate(self, decay_rate=0.75):
        """Sets the learning rate to the initial LR decayed by 'decay_rate'"""

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        old_lr = get_lr(self.optimizer)
        assert old_lr > 0
        new_lr = decay_rate * old_lr
        self.logger.info(f'Changing learning rate from {old_lr} to {new_lr}')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _is_best_val_accuracy(self, val_accuracy):
        is_best = val_accuracy > self.best_val_accuracy
        if is_best:
            self.logger.info(
                f'Saving new best validation accuracy: {val_accuracy}')
        self.best_val_accuracy = max(val_accuracy, self.best_val_accuracy)
        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
            'max_patience': self.max_patience
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_stats(self, phase, loss_avg, accuracy_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_accuracy_avg': accuracy_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(),
                                      self.num_iterations)
            self.writer.add_histogram(name + '/grad',
                                      value.grad.data.cpu().numpy(),
                                      self.num_iterations)

    def _log_images(self, input, target, prediction):
        sources = {
            'inputs': input.data.cpu().numpy(),
            'targets': target.data.cpu().numpy(),
            'predictions': prediction.data.cpu().numpy()
        }
        for name, batch in sources.items():
            for tag, image in self._images_from_batch(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='HW')

    def _images_from_batch(self, name, batch):
        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, (self._normalize_img(img))))
        else:
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, (self._normalize_img(img))))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return (img - np.min(img)) / np.ptp(img)
