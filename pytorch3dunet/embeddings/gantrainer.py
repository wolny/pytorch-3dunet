import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.embeddings.utils import extract_instance_masks, MeanEmbeddingAnchor, RandomEmbeddingAnchor
from pytorch3dunet.unet3d.losses import get_loss_criterion, AuxContrastiveLoss, _AbstractContrastiveLoss
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import get_logger, get_number_of_learnable_parameters, create_optimizer, \
    create_lr_scheduler, get_tensorboard_formatter, create_sample_plotter, RunningAverage, save_checkpoint, \
    load_checkpoint

logger = get_logger('GANTrainer')


class EmbeddingGANTrainerBuilder:
    @staticmethod
    def build(config):
        G = get_model(config['G_model'])
        D = get_model(config['D_model'])
        # use DataParallel if more than 1 GPU available
        device = config['device']
        if torch.cuda.device_count() > 1 and not device.type == 'cpu':
            G = nn.DataParallel(G)
            D = nn.DataParallel(D)
            logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

        # put the model on GPUs
        logger.info(f"Sending the G and D to '{config['device']}'")
        G = G.to(device)
        D = D.to(device)

        # Log the number of learnable parameters
        logger.info(f'Number of learnable params G: {get_number_of_learnable_parameters(G)}')
        logger.info(f'Number of learnable params D: {get_number_of_learnable_parameters(D)}')

        # Create loss criterion
        G_loss_criterion = get_loss_criterion(config)
        # Create evaluation metric
        G_eval_criterion = get_evaluation_metric(config)

        # Create data loaders
        loaders = get_train_loaders(config)

        # Create the optimizer
        G_optimizer = create_optimizer(config['G_optimizer'], G)
        D_optimizer = create_optimizer(config['D_optimizer'], D)

        # Create learning rate adjustment strategy
        G_lr_scheduler = create_lr_scheduler(config.get('G_lr_scheduler', None), G_optimizer)

        trainer_config = config['trainer']
        # get tensorboard formatter
        tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
        # get sample plotter
        sample_plotter = create_sample_plotter(trainer_config.pop('sample_plotter', None))

        resume = trainer_config.get('resume', None)
        pre_trained = trainer_config.get('pre_trained', None)

        if resume is not None:
            logger.info(f'Resuming training from: {resume}')
            return EmbeddingGANTrainer.from_checkpoint(
                G=G,
                D=D,
                G_optimizer=G_optimizer,
                D_optimizer=D_optimizer,
                G_lr_scheduler=G_lr_scheduler,
                G_loss_criterion=G_loss_criterion,
                G_eval_criterion=G_eval_criterion,
                device=device,
                loaders=loaders,
                tensorboard_formatter=tensorboard_formatter,
                sample_plotter=sample_plotter,
                **trainer_config
            )
        elif pre_trained is not None:
            logger.info(f'Using pretrained embedding model: {pre_trained}')
            return EmbeddingGANTrainer.from_pretrained_emb(
                G=G,
                D=D,
                G_optimizer=G_optimizer,
                D_optimizer=D_optimizer,
                G_lr_scheduler=G_lr_scheduler,
                G_loss_criterion=G_loss_criterion,
                G_eval_criterion=G_eval_criterion,
                device=device,
                loaders=loaders,
                tensorboard_formatter=tensorboard_formatter,
                sample_plotter=sample_plotter,
                **trainer_config
            )
        else:
            # Create model trainer
            return EmbeddingGANTrainer(
                G=G,
                D=D,
                G_optimizer=G_optimizer,
                D_optimizer=D_optimizer,
                G_lr_scheduler=G_lr_scheduler,
                G_loss_criterion=G_loss_criterion,
                G_eval_criterion=G_eval_criterion,
                device=device,
                loaders=loaders,
                tensorboard_formatter=tensorboard_formatter,
                sample_plotter=sample_plotter,
                **trainer_config
            )


class EmbeddingGANTrainer:
    def __init__(self, G, D, G_optimizer, D_optimizer, G_lr_scheduler, G_loss_criterion, G_eval_criterion,
                 device, loaders, checkpoint_dir, gan_loss_weight, D_iters=500, combine_masks=False,
                 anchor_extraction='mean', label_smoothing=True,
                 max_num_epochs=100, max_num_iterations=int(1e5), validate_after_iters=2000, log_after_iters=500,
                 num_iterations=1, num_epoch=0, eval_score_higher_is_better=True,
                 best_eval_score=None, tensorboard_formatter=None, sample_plotter=None,
                 **kwargs):
        self.sample_plotter = sample_plotter
        self.tensorboard_formatter = tensorboard_formatter
        self.best_eval_score = best_eval_score
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.num_epoch = num_epoch
        self.num_iterations = num_iterations
        self.log_after_iters = log_after_iters
        self.validate_after_iters = validate_after_iters
        self.max_num_iterations = max_num_iterations
        self.max_num_epochs = max_num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.loaders = loaders
        self.device = device
        self.G_eval_criterion = G_eval_criterion
        self.G_loss_criterion = G_loss_criterion
        self.G_lr_scheduler = G_lr_scheduler
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.D = D
        self.G = G
        self.gan_loss_weight = gan_loss_weight
        self.D_iters = D_iters
        self.combine_masks = combine_masks
        assert anchor_extraction in ['mean', 'random']
        if anchor_extraction == 'mean':
            assert isinstance(self.G_loss_criterion, _AbstractContrastiveLoss)
            # function for computing a mean embeddings of target instances
            c_mean_fn = self.G_loss_criterion._compute_cluster_means
            self.anchor_embeddings_extrator = MeanEmbeddingAnchor(c_mean_fn)
        else:
            self.anchor_embeddings_extrator = RandomEmbeddingAnchor()
        self.label_smoothing = label_smoothing

        logger.info('GENERATOR')
        logger.info(self.G)
        logger.info('DISCRIMINATOR')
        logger.info(self.D)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')
        logger.info(f'D_iters: {D_iters}, gan_loss_weight: {gan_loss_weight}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        # hardcode pmaps_threshold for now
        self.dist_to_mask = AuxContrastiveLoss.Gaussian(G_loss_criterion.delta_var, pmaps_threshold=0.5)
        # use BCELoss for the GAN discriminator
        self.bce_loss = nn.BCELoss()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, G, D, G_optimizer, D_optimizer, G_lr_scheduler, G_loss_criterion,
                        G_eval_criterion, loaders, tensorboard_formatter=None, sample_plotter=None, **kwargs):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = load_checkpoint(checkpoint_path, G, G_optimizer)
        _ = load_checkpoint(checkpoint_path, D, D_optimizer, model_key='D_model_state_dict',
                            optimizer_key='D_optimizer_state_dict')
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(G, D, G_optimizer, D_optimizer, G_lr_scheduler,
                   G_loss_criterion, G_eval_criterion,
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
                   skip_train_validation=state.get('skip_train_validation', False),
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter,
                   **kwargs)

    @classmethod
    def from_pretrained_emb(cls, pre_trained, G, D, G_optimizer, D_optimizer, G_lr_scheduler, G_loss_criterion,
                            G_eval_criterion, loaders, tensorboard_formatter=None, sample_plotter=None, **kwargs):
        logger.info(f"Loading checkpoint '{pre_trained}'...")
        state = load_checkpoint(pre_trained, G)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        return cls(G, D, G_optimizer, D_optimizer, G_lr_scheduler,
                   G_loss_criterion, G_eval_criterion,
                   kwargs.pop('device'),
                   loaders,
                   checkpoint_dir=kwargs.pop('checkpoint_dir'),
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter,
                   **kwargs)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epoch += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        emb_losses = RunningAverage()
        G_losses = RunningAverage()
        D_losses = RunningAverage()
        G_eval_scores = RunningAverage()

        # sets the model in training mode
        self.G.train()
        self.D.train()

        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            input, target = self._split_training_batch(t)
            # forward pass through embedding network (generator)
            output = self.G(input)

            if (self.num_iterations // self.D_iters) % 2 == 0:
                # train discriminator
                self._unfreeze_D()

                # create real and fake masks
                output_det = output.detach()  # make sure that G is not updated
                real_masks, fake_masks = extract_instance_masks(output_det, target,
                                                                self.anchor_embeddings_extrator,
                                                                self.dist_to_mask,
                                                                self.combine_masks,
                                                                self.label_smoothing)
                if real_masks is None or fake_masks is None:
                    # skip background patches
                    continue

                # create mask labels for the discriminator
                real_labels = torch.ones(real_masks.size(0), 1).to(self.device)
                fake_labels = torch.zeros(fake_masks.size(0), 1).to(self.device)

                # compute BCE loss using real masks
                D_real = self.D(real_masks)
                D_real_loss = self.bce_loss(D_real, real_labels)

                # compute BCE loss using fake masks
                D_fake = self.D(fake_masks)
                D_fake_loss = self.bce_loss(D_fake, fake_labels)

                # optimize discriminator
                D_loss = D_real_loss + D_fake_loss
                self.D_optimizer.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

                D_losses.update(D_loss, real_masks.size(0) + fake_masks.size(0))
            else:
                # train generator
                self._freeze_D()

                self.G_optimizer.zero_grad()

                _, fake_masks = extract_instance_masks(output, target,
                                                       self.anchor_embeddings_extrator,
                                                       self.dist_to_mask,
                                                       self.combine_masks,
                                                       self.label_smoothing)
                if fake_masks is None:
                    # train only with embedding loss if only background is present
                    emb_loss = self.G_loss_criterion(output, target)
                    emb_loss.backward()
                    self.G_optimizer.step()
                    emb_losses.update(emb_loss.item(), self._batch_size(input))
                    continue

                # compute embedding loss
                emb_loss = self.G_loss_criterion(output, target)
                # emb_loss.backward(retain_graph=True)
                emb_losses.update(emb_loss.item(), self._batch_size(input))

                # compute adversarial loss using fake images
                real_labels = torch.ones(fake_masks.size(0), 1).to(self.device)
                outputs = self.D(fake_masks)
                G_loss = self.bce_loss(outputs, real_labels)
                G_losses.update(G_loss, real_labels.size(0))

                # combined embedding and G_loss
                combined_loss = emb_loss + self.gan_loss_weight * G_loss
                # optimize generator
                combined_loss.backward()
                self.G_optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.G.eval()
                # evaluate on validation set
                eval_score = self.validate()
                # set the model back to training mode
                self.G.train()

                # adjust learning rate if necessary
                if self.G_lr_scheduler is not None:
                    if isinstance(self.G_lr_scheduler, ReduceLROnPlateau):
                        self.G_lr_scheduler.step(eval_score)
                    else:
                        self.G_lr_scheduler.step()
                # log current learning rate in tensorboard
                self._log_G_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                eval_score = self.G_eval_criterion(output, target)
                G_eval_scores.update(eval_score.item(), self._batch_size(input))

                # log stats, params and images
                logger.info(
                    f'Training stats. Embedding Loss: {emb_losses.avg}. GAN Loss: {G_losses.avg}. '
                    f'Discriminator Loss: {D_losses.avg}. Evaluation score: {G_eval_scores.avg}')

                self.writer.add_scalar('train_embedding_loss', emb_losses.avg, self.num_iterations)
                if (self.num_iterations // self.D_iters) % 2 == 0:
                    # discriminator phase
                    self.writer.add_scalar('train_D_loss', D_losses.avg, self.num_iterations)
                    # log images
                    inputs_map = {
                        'inputs': input,
                        'targets': target,
                        'predictions': output,
                        'real_masks': real_masks,
                        'fake_masks': fake_masks

                    }
                    self._log_images(inputs_map)

                else:
                    self.writer.add_scalar('train_GAN_loss', G_losses.avg, self.num_iterations)

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
        lr = self.G_optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = RunningAverage()
        val_scores = RunningAverage()

        if self.sample_plotter is not None:
            self.sample_plotter.update_current_dir()

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, target = self._split_training_batch(t)

                output = self.G(input)
                loss = self.G_loss_criterion(output, target)
                val_losses.update(loss.item(), self._batch_size(input))

                eval_score = self.G_eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

                if self.sample_plotter is not None:
                    self.sample_plotter(i, input, output, target, 'val')

            self.writer.add_scalar('val_embedding_loss', val_losses.avg, self.num_iterations)
            self.writer.add_scalar('val_eval', val_scores.avg, self.num_iterations)
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        input, target = _move_to_device(t)
        return input, target

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
        if isinstance(self.G, nn.DataParallel):
            G_state_dict = self.G.module.state_dict()
            D_state_dict = self.D.module.state_dict()
        else:
            G_state_dict = self.G.state_dict()
            D_state_dict = self.D.state_dict()

        # save generator and discriminator state + metadata
        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': G_state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.G_optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            # discriminator
            'D_model_state_dict': D_state_dict,
            'D_optimizer_state_dict': self.D_optimizer.state_dict()
        },
            is_best=is_best,
            checkpoint_dir=self.checkpoint_dir,
            logger=logger)

    def _log_G_lr(self):
        lr = self.G_optimizer.param_groups[0]['lr']
        self.writer.add_scalar('G_learning_rate', lr, self.num_iterations)

    def _log_images(self, inputs_map):
        assert isinstance(inputs_map, dict)
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    def _freeze_D(self):
        for p in self.D.parameters():
            p.requires_grad = False

    def _unfreeze_D(self):
        for p in self.D.parameters():
            p.requires_grad = True
