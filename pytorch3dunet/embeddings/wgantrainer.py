import os

import torch
from torch import autograd
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch3dunet.embeddings.utils import extract_instance_masks, AbstractEmbeddingGANTrainerBuilder, \
    AbstractEmbeddingGANTrainer
from pytorch3dunet.unet3d.utils import get_logger, RunningAverage, load_checkpoint

logger = get_logger('WGANTrainer')


class EmbeddingWGANTrainerBuilder(AbstractEmbeddingGANTrainerBuilder):
    @staticmethod
    def trainer_class():
        return EmbeddingWGANTrainer


class EmbeddingWGANTrainer(AbstractEmbeddingGANTrainer):
    def __init__(self, G, D, G_optimizer, D_optimizer, G_lr_scheduler, G_loss_criterion, G_eval_criterion, device,
                 loaders, checkpoint_dir, gp_lambda, gan_loss_weight, critic_iters, combine_masks=False,
                 anchor_extraction='mean', label_smoothing=True, max_num_epochs=100, max_num_iterations=int(1e5),
                 validate_after_iters=2000, log_after_iters=500, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None, tensorboard_formatter=None,
                 sample_plotter=None, **kwargs):

        super().__init__(G, D, G_optimizer, D_optimizer, G_lr_scheduler, G_loss_criterion, G_eval_criterion,
                         gan_loss_weight, device, loaders, checkpoint_dir, combine_masks, anchor_extraction,
                         label_smoothing, max_num_epochs, max_num_iterations, validate_after_iters, log_after_iters,
                         num_iterations, num_epoch, eval_score_higher_is_better, best_eval_score, tensorboard_formatter,
                         sample_plotter, **kwargs)
        self.gp_lambda = gp_lambda
        self.critic_iters = critic_iters

    @classmethod
    def from_checkpoint(cls, checkpoint_path, G, D, G_optimizer, D_optimizer, G_lr_scheduler,
                        G_loss_criterion, G_eval_criterion, loaders,
                        tensorboard_formatter=None, sample_plotter=None, **kwargs):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        # load generator, i.e. embedding model
        state = load_checkpoint(checkpoint_path, G, optimizer=G_optimizer,
                                model_key='model_state_dict', optimizer_key='optimizer_state_dict')
        # load critic
        state = load_checkpoint(checkpoint_path, D, optimizer=D_optimizer,
                                model_key='D_model_state_dict', optimizer_key='D_optimizer_state_dict')
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(G, D, G_optimizer, D_optimizer, G_lr_scheduler, G_loss_criterion, G_eval_criterion,
                   torch.device(state['device']), loaders, checkpoint_dir,
                   kwargs['gp_lambda'], kwargs['gan_loss_weight'], kwargs['critic_iters'], kwargs['combine_masks'],
                   kwargs['anchor_extraction'], kwargs['label_smoothing'],
                   max_num_epochs=state['max_num_epochs'], max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'], log_after_iters=state['log_after_iters'],
                   num_iterations=state['num_iterations'], num_epoch=state['epoch'],
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'], tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter)

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        # keeps running average of the contrastive loss
        emb_losses = RunningAverage()
        # keeps track of the generator part of the GAN loss
        G_losses = RunningAverage()
        # keeps track of the discriminator part of the GAN loss
        D_losses = RunningAverage()
        # keeps track of the eval score of the generator (i.e. embedding network)
        G_eval_scores = RunningAverage()
        # keeps track of the estimate of Wasserstein Distance
        Wasserstein_dist = RunningAverage()

        # sets the model in training mode
        self.G.train()
        self.D.train()

        one = torch.FloatTensor([1])
        one = one.to(self.device)
        mone = one * -1

        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            input, target = self.split_training_batch(t)

            if self.num_iterations % self._D_iters() == 0:
                # update G network
                self.freeze_D()
                self.G_optimizer.zero_grad()

                # forward pass through embedding network (generator)
                output = self.G(input)

                # compute embedding loss
                emb_loss = self.G_loss_criterion(output, target)
                emb_losses.update(emb_loss.item(), self.batch_size(input))

                # compute GAN loss
                # real_masks are not used in the G update phase, but are needed for tensorboard logging later
                real_masks, fake_masks = extract_instance_masks(output, target,
                                                                self.anchor_embeddings_extractor,
                                                                self.dist_to_mask,
                                                                self.combine_masks,
                                                                self.label_smoothing)
                if fake_masks is None:
                    # skip background patches and backprop only through embedding loss
                    emb_loss.backward()
                    self.G_optimizer.step()
                    continue

                G_loss = self.D(fake_masks)
                G_loss = G_loss.mean(dim=0)
                G_losses.update(-G_loss.item(), self.batch_size(fake_masks))

                # compute combined embedding and GAN loss; make sure to minimize -G_loss
                combined_loss = emb_loss - self.gan_loss_weight * G_loss
                combined_loss.backward()

                self.G_optimizer.step()

                self.unfreeze_D()

                self.G_iterations += 1

                if self.G_iterations % self.log_after_iters == 0:
                    logger.info('Logging params and gradients of G')
                    # log params and gradients for G only cause D is frozen
                    self.log_params(self.G, 'G')
            else:
                # update D netowrk
                self.D_optimizer.zero_grad()

                with torch.no_grad():
                    # forward pass through embedding network (generator)
                    # make sure the G is frozen
                    output = self.G(input)

                output = output.detach()  # make sure that G is not updated

                # create real and fake instance masks
                real_masks, fake_masks = extract_instance_masks(output, target,
                                                                self.anchor_embeddings_extractor,
                                                                self.dist_to_mask,
                                                                self.combine_masks,
                                                                self.label_smoothing)

                if real_masks is None or fake_masks is None:
                    # skip background patches
                    continue

                if real_masks.size()[0] >= 40:
                    # skip if there are too many instances in the patch in order to prevent CUDA OOM errors
                    continue

                # train D with real
                D_real = self.D(real_masks)
                # average critic output across batch
                D_real = D_real.mean(dim=0)
                D_real.backward(mone)

                # train D with fake
                D_fake = self.D(fake_masks)
                # average critic output across batch
                D_fake = D_fake.mean(dim=0)
                D_fake.backward(one)

                # train with gradient penalty
                gp = self._calc_gp(real_masks, fake_masks)
                gp.backward()

                D_cost = D_fake - D_real + gp
                Wasserstein_D = D_real - D_fake

                # update D's weights
                self.D_optimizer.step()

                n_batch = 2 * self.batch_size(fake_masks)
                D_losses.update(D_cost.item(), n_batch)
                Wasserstein_dist.update(Wasserstein_D.item(), n_batch)

                self.D_iterations += 1

                if self.D_iterations % self.log_after_iters == 0:
                    # log params and gradients for D only cause G is frozen
                    logger.info('Logging params and gradients of D')
                    self.log_params(self.D, 'D')

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
                self.log_G_lr()
                # remember best validation metric
                is_best = self.is_best_eval_score(eval_score)

                # save checkpoint
                self.save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                eval_score = self.G_eval_criterion(output, target)
                G_eval_scores.update(eval_score.item(), self.batch_size(input))

                # log stats, params and images
                logger.info(
                    f'Training stats. Embedding Loss: {emb_losses.avg}. GAN Loss: {G_losses.avg}. '
                    f'Discriminator Loss: {D_losses.avg}. Evaluation score: {G_eval_scores.avg}')

                self.writer.add_scalar('train_embedding_loss', emb_losses.avg, self.num_iterations)
                self.writer.add_scalar('train_GAN_loss', G_losses.avg, self.num_iterations)
                self.writer.add_scalar('train_D_loss', D_losses.avg, self.num_iterations)
                self.writer.add_scalar('Wasserstein_distance', Wasserstein_dist.avg, self.num_iterations)

                inputs_map = {
                    'inputs': input,
                    'targets': target,
                    'predictions': output
                }
                self.log_images(inputs_map)
                # log masks if we're not during G training phase
                if self.num_iterations % (self.critic_iters + 1) != 0:
                    inputs_map = {
                        'real_masks': real_masks,
                        'fake_masks': fake_masks
                    }
                    self.log_images(inputs_map)

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def _D_iters(self):
        # make sure Discriminator is trained more at the beginning
        if self.G_iterations < 25:
            return 100
        return self.critic_iters + 1

    def _calc_gp(self, real_masks, fake_masks):
        n_batch = real_masks.size(0)

        alpha = torch.rand(n_batch, 1, 1, 1, 1)
        alpha = alpha.expand_as(real_masks)
        alpha = alpha.to(real_masks.device)

        interpolates = alpha * real_masks + ((1 - alpha) * fake_masks)
        interpolates.requires_grad = True

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(real_masks.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        return gradient_penalty
