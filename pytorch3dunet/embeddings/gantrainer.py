import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.losses import get_loss_criterion, AuxContrastiveLoss
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import get_logger, get_number_of_learnable_parameters, create_optimizer, \
    create_lr_scheduler, get_tensorboard_formatter, get_sample_plotter, RunningAverage, save_checkpoint, \
    expand_as_one_hot

logger = get_logger('WGANTrainer')


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
        sample_plotter = get_sample_plotter(trainer_config.pop('sample_plotter', None))

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
                 device, loaders, checkpoint_dir, gan_loss_weight,
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

        logger.info('GENERATOR')
        logger.info(self.G)
        logger.info('DISCRIMINATOR')
        logger.info(self.D)
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

        # hardcode pmaps_threshold for now
        self.dist_to_mask = AuxContrastiveLoss.Gaussian(G_loss_criterion.delta_var, pmaps_threshold=0.5)
        # use BCELoss for the GAN discriminator
        self.bce_loss = nn.BCELoss()

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

            input, target, _ = self._split_training_batch(t)
            # forward pass through embedding network (generator)
            output = self.G(input)

            if self.num_iterations % 2 == 0:
                # train discriminator

                # create real and fake masks
                output_det = output.detach()  # make sure that G is not updated
                real_masks, fake_masks = self._extract_instance_masks(output_det, target)
                if real_masks is None or fake_masks is None:
                    # skip background patches
                    continue

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

                D_losses.update(D_loss, 2 * real_masks.size(0))
            else:
                # train generator
                self._freeze_D()

                self.G_optimizer.zero_grad()

                _, fake_masks = self._extract_instance_masks(output, target)
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

                # compute loss with fake masks
                real_labels = torch.ones(fake_masks.size(0), 1).to(self.device)
                outputs = self.D(fake_masks)
                G_loss = self.bce_loss(outputs, real_labels)
                G_losses.update(G_loss, real_labels.size(0))

                # combined embedding and G_loss
                combined_loss = emb_loss + self.gan_loss_weight * G_loss
                # optimize generator
                combined_loss.backward()
                self.G_optimizer.step()

                self._unfreeze_D()

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
                self.writer.add_scalar('train_GAN_loss', G_losses.avg, self.num_iterations)
                self.writer.add_scalar('train_D_loss', D_losses.avg, self.num_iterations)

                inputs_map = {
                    'inputs': input,
                    'targets': target,
                    'predictions': output
                }
                self._log_images(inputs_map)
                # log masks if we're not during G training phase
                if self.num_iterations % 2 == 0:
                    inputs_map = {
                        'real_masks': real_masks,
                        'fake_masks': fake_masks
                    }
                    self._log_images(inputs_map)

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

                input, target, _ = self._split_training_batch(t)

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

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

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
            state_dict = self.G.module.state_dict()
        else:
            state_dict = self.G.state_dict()

        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.G_optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
        }, is_best, checkpoint_dir=self.checkpoint_dir,
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

    def _extract_instance_masks(self, embeddings, target):
        # iterate over batch
        real_masks = []
        fake_masks = []

        for emb, tar in zip(embeddings, target):
            cluster_means = self._compute_cluster_means(emb, tar)
            for i, cm in enumerate(cluster_means):
                if i == 0:
                    # ignore 0-label
                    continue

                # compute distance map; embeddings is ExSPATIAL, cluster_mean is ExSINGLETON_SPATIAL, so we can just broadcast
                dist_to_mean = torch.norm(emb - cm, 'fro', dim=0)
                # convert distance map to instance pmaps
                inst_pmap = self.dist_to_mask(dist_to_mean)
                # add channel dim
                fake_masks.append(inst_pmap.unsqueeze(0))

                assert i in target
                inst_mask = (tar == i).float()
                # add noise to instance_mask
                uniform_noise = torch.randn(inst_mask.size()).to(inst_mask.device) * 0.05
                inst_mask += uniform_noise
                real_masks.append(inst_mask.unsqueeze(0))

        if len(real_masks) == 0:
            return None, None

        real_masks = torch.stack(real_masks).to(embeddings.device)
        real_masks.clamp_(0, 1)
        fake_masks = torch.stack(fake_masks).to(embeddings.device)
        return real_masks, fake_masks

    def _compute_cluster_means(self, emb, tar):
        instances = torch.unique(tar)
        C = instances.size(0)

        single_target = expand_as_one_hot(tar.unsqueeze(0), C).squeeze(0)
        single_target = single_target.unsqueeze(1)
        spatial_dims = emb.dim() - 1

        cluster_means, _, _ = self.G_loss_criterion._compute_cluster_means(emb, single_target, spatial_dims)
        return cluster_means
