import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch3dunet.embeddings.utils import extract_instance_masks, AbstractEmbeddingGANTrainerBuilder, \
    AbstractEmbeddingGANTrainer
from pytorch3dunet.unet3d.utils import get_logger, RunningAverage, load_checkpoint

logger = get_logger('GANTrainer')


class EmbeddingGANTrainerBuilder(AbstractEmbeddingGANTrainerBuilder):
    @staticmethod
    def trainer_class():
        return EmbeddingGANTrainer


class EmbeddingGANTrainer(AbstractEmbeddingGANTrainer):
    def __init__(self, G, D, G_optimizer, D_optimizer, G_lr_scheduler, G_loss_criterion, G_eval_criterion, device,
                 loaders, checkpoint_dir, gan_loss_weight, D_iters=500, combine_masks=False, anchor_extraction='mean',
                 label_smoothing=True, max_num_epochs=100, max_num_iterations=int(1e5), validate_after_iters=2000,
                 log_after_iters=500, num_iterations=1, num_epoch=0, eval_score_higher_is_better=True,
                 best_eval_score=None, tensorboard_formatter=None, sample_plotter=None, **kwargs):

        super().__init__(G, D, G_optimizer, D_optimizer, G_lr_scheduler, G_loss_criterion, G_eval_criterion,
                         gan_loss_weight, device, loaders, checkpoint_dir, combine_masks, anchor_extraction,
                         label_smoothing, max_num_epochs, max_num_iterations, validate_after_iters, log_after_iters,
                         num_iterations, num_epoch, eval_score_higher_is_better, best_eval_score, tensorboard_formatter,
                         sample_plotter, **kwargs)

        self.D_iters = D_iters
        logger.info(f'D_iters: {D_iters}')

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

            input, target = self.split_training_batch(t)
            # forward pass through embedding network (generator)
            output = self.G(input)

            if (self.num_iterations // self.D_iters) % 2 == 0:
                # train discriminator
                self.unfreeze_D()

                # create real and fake masks
                output_det = output.detach()  # make sure that G is not updated
                real_masks, fake_masks = extract_instance_masks(output_det, target,
                                                                self.anchor_embeddings_extractor,
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
                self.freeze_D()

                self.G_optimizer.zero_grad()

                _, fake_masks = extract_instance_masks(output, target,
                                                       self.anchor_embeddings_extractor,
                                                       self.dist_to_mask,
                                                       self.combine_masks,
                                                       self.label_smoothing)
                if fake_masks is None:
                    # train only with embedding loss if only background is present
                    emb_loss = self.G_loss_criterion(output, target)
                    emb_loss.backward()
                    self.G_optimizer.step()
                    emb_losses.update(emb_loss.item(), self.batch_size(input))
                    continue

                # compute embedding loss
                emb_loss = self.G_loss_criterion(output, target)
                # emb_loss.backward(retain_graph=True)
                emb_losses.update(emb_loss.item(), self.batch_size(input))

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
                self.log_G_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                eval_score = self.G_eval_criterion(output, target)
                G_eval_scores.update(eval_score.item(), self.batch_size(input))

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
                    self.log_images(inputs_map)

                else:
                    self.writer.add_scalar('train_GAN_loss', G_losses.avg, self.num_iterations)

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False
