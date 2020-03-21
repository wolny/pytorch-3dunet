import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch3dunet.unet3d.utils import get_logger
from . import utils

logger = get_logger('NetworkTrainer')


class LearningCurves:
    """
    Keeps track of the learning curves.
    """

    def __init__(self):
        self.losses = utils.RunningAverage()
        self.eval_scores = utils.RunningAverage()

    def update_loss(self, loss, batch_size):
        self.losses.update(loss, batch_size)

    def update_eval(self, eval, batch_size):
        self.eval_scores.update(eval, batch_size)

    @property
    def loss_avg(self):
        return self.losses.avg

    @property
    def eval_avg(self):
        return self.eval_scores.avg


class AbstractTrainer:
    """Abstract class responsible for model training.

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
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        sample_plotter (callable): saves sample inputs, network outputs and targets to a given directory
            during validation phase
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs, max_num_iterations,
                 validate_after_iters, log_after_iters,
                 validate_iters, num_iterations, num_epoch,
                 eval_score_higher_is_better, best_eval_score,
                 tensorboard_formatter, sample_plotter,
                 skip_train_validation, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.sample_plotter = sample_plotter
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        tensorboard_formatter=None, sample_plotter=None, skip_train_validation=False):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter,
                   skip_train_validation=skip_train_validation)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=1e5,
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        tensorboard_formatter=None, sample_plotter=None,
                        skip_train_validation=False):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        utils.load_checkpoint(pre_trained, model, None)
        checkpoint_dir = os.path.split(pre_trained)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter,
                   skip_train_validation=skip_train_validation)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epoch += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def forward_backward_pass(self, batch, learning_curves):
        """
        Perform one training iteration. To be implemented by a concrete implementation of the trainer.

        Args:
            batch (tuple): tuple of torch tensors containing input and target
            learning_curves (LearningCurves): keeps track of the loss and eval score

        Returns:
            tuple: input, output, target
        """
        raise NotImplementedError

    def val_forward_pass(self, batch, learning_curves):
        """
        Perform the forward pass during the validation phase.

        Returns:
             tuple: input, output, target
        """
        raise NotImplementedError

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            `True` if the training should be terminated immediately, `False` otherwise
        """

        # create object to keep track of learning curves during training phase
        train_curves = self.create_learning_curves()

        # sets the model in training mode
        self.model.train()

        for t in self.loaders['train']:
            logger.info(
                f'Training iteration {self.num_iterations}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            # make forward pass and backprop
            input, output, target = self.forward_backward_pass(t, train_curves)

            if self.num_iterations % self.validate_after_iters == 0:
                self.validate_and_save()

            if self.num_iterations % self.log_after_iters == 0:
                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric as well as images in tensorboard will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    output = self.model.final_activation(output)

                # compute eval criterion
                if not self.skip_train_validation:
                    eval_score = self.eval_criterion(output, target)
                    train_curves.update_eval(eval_score.item(), self._batch_size(input))

                # log stats, params and images
                logger.info(
                    f'Training stats. Loss: {train_curves.loss_avg}. Evaluation score: {train_curves.eval_avg}')
                self._log_stats('train', train_curves.loss_avg, train_curves.eval_avg)
                self._log_params()
                self._log_images(input, target, output, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def validate_and_save(self):
        """
        Computes stats on the validation set, updates learning rate if necessary and saves checkpoints
        """
        # set the model in eval mode
        self.model.eval()
        # evaluate on validation set
        eval_score = self.validate()
        # set the model back to training mode
        self.model.train()
        if self.scheduler is not None:
            # adjust learning rate if necessary
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(eval_score)
            else:
                self.scheduler.step()
        # log current learning rate in tensorboard
        self._log_lr()
        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score)
        # save last checkpoint after every validation phase; save best checkpoint only if is_best==true
        self._save_checkpoint(is_best)

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

        val_curves = self.create_learning_curves()

        if self.sample_plotter is not None:
            self.sample_plotter.update_current_dir()

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, output, target = self.val_forward_pass(t, val_curves)

                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    output = self.model.final_activation(output)

                if i % 100 == 0:
                    self._log_images(input, target, output, 'val_')

                eval_score = self.eval_criterion(output, target)
                val_curves.update_eval(eval_score.item(), self._batch_size(input))

                if self.sample_plotter is not None:
                    self.sample_plotter(i, input, output, target, 'val')

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            self._log_stats('val', val_curves.loss_avg, val_curves.eval_avg)
            logger.info(f'Validation finished. Loss: {val_curves.loss_avg}. Evaluation score: {val_curves.eval_avg}')
            return val_curves.eval_avg

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
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

    def _log_images(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    @staticmethod
    def create_learning_curves():
        return LearningCurves()


class UNet3DTrainer(AbstractTrainer):
    def __init__(self, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=1e5, validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0, eval_score_higher_is_better=True,
                 best_eval_score=None, tensorboard_formatter=None, sample_plotter=None, skip_train_validation=False,
                 **kwargs):
        super().__init__(model, optimizer, lr_scheduler, loss_criterion, eval_criterion, device, loaders,
                         checkpoint_dir, max_num_epochs, max_num_iterations, validate_after_iters, log_after_iters,
                         validate_iters, num_iterations, num_epoch, eval_score_higher_is_better, best_eval_score,
                         tensorboard_formatter, sample_plotter, skip_train_validation, **kwargs)

    def forward_backward_pass(self, batch, learning_curves):
        input, output, target, loss = self._forward_and_update(batch, learning_curves)

        # compute gradients and update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return input, output, target

    def val_forward_pass(self, batch, learning_curves):
        input, output, target, _ = self._forward_and_update(batch, learning_curves)
        return input, output, target

    def _forward_and_update(self, batch, learning_curves):
        input, target, weight = self._split_training_batch(batch)
        output, loss = self._forward_pass(input, target, weight)
        learning_curves.update_loss(loss.item(), self._batch_size(input))
        return input, output, target, loss

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight


class EmbeddingWGANTrainer(AbstractTrainer):
    def __init__(self, G, G_optimizer, G_scheduler, emb_loss, eval_criterion,
                 device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=1e5, validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0, eval_score_higher_is_better=True,
                 best_eval_score=None, tensorboard_formatter=None, sample_plotter=None, skip_train_validation=False,
                 **kwargs):
        super().__init__(G, G_optimizer, G_scheduler, emb_loss, eval_criterion, device, loaders,
                         checkpoint_dir, max_num_epochs, max_num_iterations, validate_after_iters, log_after_iters,
                         validate_iters, num_iterations, num_epoch, eval_score_higher_is_better, best_eval_score,
                         tensorboard_formatter, sample_plotter, skip_train_validation, **kwargs)
        self.D = None
        self.D_optimizer = None

    def forward_backward_pass(self, batch, learning_curves):
        # pass iteration number
        pass

    def val_forward_pass(self, batch, learning_curves):
        # used only during the validation
        pass
