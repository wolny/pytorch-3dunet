import torch
from torch import autograd
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch3dunet.embeddings.utils import extract_instance_masks, AbstractEmbeddingGANTrainerBuilder, \
    add_gaussian_noise
from pytorch3dunet.embeddings.wgantrainer import EmbeddingWGANTrainer
from pytorch3dunet.unet3d.utils import get_logger, RunningAverage

logger = get_logger('WGANTrainer')


class DAEmbeddingWGANTrainerBuilder(AbstractEmbeddingGANTrainerBuilder):
    @staticmethod
    def trainer_class():
        return DAEmbeddingWGANTrainer


def extract_real_masks(target, label_smoothing):
    real_masks = []

    for tar in target:
        rms = []
        for i in torch.unique(tar):
            inst_mask = (tar == i).float()
            if label_smoothing:
                # add noise to instance mask to prevent discriminator from converging too quickly
                inst_mask = add_gaussian_noise(inst_mask)
                # clamp values
                inst_mask.clamp_(0, 1)

            # add channel dim and save real masks
            rms.append(inst_mask.unsqueeze(0))

        real_masks.extend(rms)

    real_masks = torch.stack(real_masks).to(target.device)
    return real_masks


def extract_fake_masks(embeddings, dist_to_mask_fn, volume_threshold=0.01, max_instances=40):
    fake_masks = []

    for emb in embeddings:
        visited = torch.ones(emb.shape[1:])

        fms = []
        fms_sizes = []
        while visited.sum() > visited.numel() * volume_threshold:
            z, y, x = torch.nonzero(visited, as_tuple=True)
            ind = torch.randint(len(z), (1,))[0]
            anchor_emb = emb[:, z[ind], y[ind], x[ind]]
            # (E,) -> (E, 1, 1, 1)
            anchor_emb = anchor_emb[..., None, None, None]

            # compute distance map; embeddings is ExSPATIAL, anchor_embeddings is ExSINGLETON_SPATIAL, so we can just broadcast
            dist_to_anchor = torch.norm(emb - anchor_emb, 'fro', dim=0)
            # TODO: get the threshold as a dist_var from the Contrastive Loss
            inst_mask = dist_to_anchor < 0.5
            # convert distance map to instance pmaps
            inst_pmap = dist_to_mask_fn(dist_to_anchor)

            fms_sizes.append(inst_mask.sum())
            fms.append(inst_pmap.unsqueeze(0))

            # update visited array
            visited[inst_mask] = 0

        # get the biggest instances and limit the instances due to OOM errors
        fms = [x for _, x in sorted(zip(fms_sizes, fms), key=lambda pair: pair[0])]
        fms = fms[:max_instances]

        fake_masks.extend(fms)

    fake_masks = torch.stack(fake_masks).to(embeddings.device)
    return fake_masks


class DAEmbeddingWGANTrainer(EmbeddingWGANTrainer):
    """
    Embedding loss on the source domain, GAN loss on the source and target domains.
    """

    def split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                if isinstance(input, torch.Tensor):
                    return input.to(self.device)
                else:
                    return input

        if len(t) == 3:
            # training phase
            input, target, domain = _move_to_device(t)
            return input, target, domain
        elif len(t) == 2:
            input, target = _move_to_device(t)
            return input, target
        else:
            raise RuntimeError(f'Incorrect tuple size: {len(t)}')

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
        # keeps track of the discriminator loss for the source and target domains
        D_fake_cost_source = RunningAverage()
        D_fake_cost_target = RunningAverage()
        D_real_cost = RunningAverage()
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

            input, target, domain = self.split_training_batch(t)
            # 0 - source domain, 1 - target domain
            # FIXME: in order to allow batches of size > 1 make sure that the batch comes from a single domain
            assert self.batch_size(input) == 1, 'Only batch size of 1 supported for now'

            if self.num_iterations % self._D_iters() == 0:
                logger.info(f'G; domain: {domain.item()}')
                # update G network
                self.freeze_D()
                self.G_optimizer.zero_grad()

                # forward pass through embedding network (generator)
                output = self.G(input)

                # compute real and fake masks
                # real_masks are not used in the G update phase, but are needed for tensorboard logging later
                real_masks = extract_real_masks(target, self.label_smoothing)
                fake_masks = extract_fake_masks(output, self.dist_to_mask)

                if domain == 0:
                    # compute embedding loss
                    emb_loss = self.G_loss_criterion(output, target)
                    emb_losses.update(emb_loss.item(), self.batch_size(input))
                    if fake_masks is None:
                        # skip background patches and backprop only through embedding loss
                        emb_loss.backward()
                        self.G_optimizer.step()
                        continue
                else:
                    emb_loss = 0
                    if fake_masks is None:
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
                logger.info(f'D; domain: {domain.item()}')
                # update D netowrk
                self.D_optimizer.zero_grad()

                with torch.no_grad():
                    # forward pass through embedding network (generator)
                    # make sure the G is frozen
                    output = self.G(input)

                output = output.detach()  # make sure that G is not updated

                real_masks = extract_real_masks(target, self.label_smoothing)
                fake_masks = extract_fake_masks(output, self.dist_to_mask)

                if real_masks is None or fake_masks is None:
                    # skip background patches
                    continue
                if real_masks.size()[0] >= 40:
                    # skip if there are too many instances in the patch in order to prevent CUDA OOM errors
                    continue

                # real masks should come from the source domain, loader will take care of that
                D_real = self.D(real_masks)
                # average critic output across batch
                D_real = D_real.mean(dim=0)
                D_real.backward(mone)
                # update real costs
                D_real_cost.update(D_real.item(), self.batch_size(real_masks))

                # train D with fake masks no matter if we're in real or fake
                D_fake = self.D(fake_masks)
                # average critic output across batch
                D_fake = D_fake.mean(dim=0)
                D_fake.backward(one)

                if domain == 0:
                    D_fake_cost_source.update(D_fake.item(), self.batch_size(fake_masks))
                else:
                    D_fake_cost_target.update(D_fake.item(), self.batch_size(fake_masks))

                # train with gradient penalty
                gp = self._calc_gp(real_masks, fake_masks)
                gp.backward()

                D_cost = D_fake - D_real + gp
                Wasserstein_D = D_real - D_fake

                # log D_cost and Wasserstein_dist only for samples from the source domain
                n_batch = 2 * self.batch_size(fake_masks)
                D_losses.update(D_cost.item(), n_batch)
                Wasserstein_dist.update(Wasserstein_D.item(), n_batch)

                # update D's weights
                self.D_optimizer.step()

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
                self.writer.add_scalar('D_real_cost', D_real_cost.avg, self.num_iterations)
                self.writer.add_scalar('D_fake_cost_source', D_fake_cost_source.avg, self.num_iterations)
                self.writer.add_scalar('D_fake_cost_target', D_fake_cost_target.avg, self.num_iterations)

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

    def _calc_gp(self, real_masks, fake_masks):
        # align real and fake masks
        n_batch = min(real_masks.size(0), fake_masks.size(0))

        real_masks = real_masks[:n_batch]
        fake_masks = fake_masks[:n_batch]

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


class DAEmbeddingWGANTrainerTargetBuilder(AbstractEmbeddingGANTrainerBuilder):
    @staticmethod
    def trainer_class():
        return DAEmbeddingWGANTargetTrainer


class DAEmbeddingWGANTargetTrainer(DAEmbeddingWGANTrainer):
    """
    Embedding loss on labeled source domain, GAN loss only on target domain
    """

    def split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                if isinstance(input, torch.Tensor):
                    return input.to(self.device)
                else:
                    return input

        if len(t) == 4:
            # training phase
            input, target, anchor_embeddings, domain = _move_to_device(t)
            return input, target, anchor_embeddings, domain
        elif len(t) == 2:
            # test phase
            input, target = _move_to_device(t)
            return input, target
        else:
            raise RuntimeError(f'Incorrect tuple size: {len(t)}')

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
        # keeps track of the discriminator loss for the source and target domains
        D_fake_cost = RunningAverage()
        D_real_cost = RunningAverage()
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

            input, target, target_1, domain = self.split_training_batch(t)
            # 0 - source domain, 1 - target domain
            # FIXME: in order to allow batches of size > 1 make sure that the batch comes from a single domain
            assert self.batch_size(input) == 1, 'Only batch size of 1 supported for now'

            if self.num_iterations % self._D_iters() == 0:
                logger.info(f'G; domain: {domain.item()}')
                # update G network
                self.freeze_D()
                self.G_optimizer.zero_grad()

                # forward pass through embedding network (generator)
                output = self.G(input)

                if domain == 0:
                    # extract masks using the source domain labels
                    target_instances = target
                else:
                    # extract masks using the target domain labels
                    target_instances = target_1

                # compute real and fake masks
                # TODO: solve anchor extraction when the target is None
                # real_masks are not used in the G update phase, but are needed for tensorboard logging later
                real_masks, fake_masks = extract_instance_masks(output, target_instances,
                                                                self.anchor_embeddings_extractor,
                                                                self.dist_to_mask,
                                                                self.combine_masks,
                                                                self.label_smoothing)

                if domain == 0:
                    # compute embedding loss
                    emb_loss = self.G_loss_criterion(output, target)
                    emb_losses.update(emb_loss.item(), self.batch_size(input))
                    if fake_masks is None:
                        # skip background patches and backprop only through embedding loss
                        emb_loss.backward()
                        self.G_optimizer.step()
                        continue

                    # skip GAN loss in the source domain
                    G_loss = 0
                else:
                    # skip embedding loss in the target domain
                    emb_loss = 0
                    if fake_masks is None:
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
                if domain == 0:
                    # GAN training only on the target domain
                    logger.info(f'D; domain: {domain.item()} skipped!')
                    continue
                else:
                    logger.info(f'D; domain: {domain.item()}')

                # update D netowrk
                self.D_optimizer.zero_grad()

                with torch.no_grad():
                    # forward pass through embedding network (generator)
                    # make sure the G is frozen
                    output = self.G(input)

                output = output.detach()  # make sure that G is not updated

                # create fake instance masks coming from the target domain
                _, fake_masks = extract_instance_masks(output, target_1,
                                                       self.anchor_embeddings_extractor,
                                                       self.dist_to_mask,
                                                       self.combine_masks,
                                                       self.label_smoothing)

                # create real instance masks coming from the *source* domain
                real_masks = self._extract_real_masks(target, self.combine_masks, self.label_smoothing)

                if real_masks is None or fake_masks is None:
                    # skip background patches
                    continue
                if real_masks.size()[0] >= 40 or fake_masks.size()[0] >= 40:
                    # skip if there are too many instances in the patch in order to prevent CUDA OOM errors
                    continue

                # real masks should come from the source domain, loader will take care of that
                D_real = self.D(real_masks)
                # average critic output across batch
                D_real = D_real.mean(dim=0)
                D_real.backward(mone)
                # update real costs
                D_real_cost.update(D_real.item(), self.batch_size(real_masks))

                # train D with fake masks no matter if we're in real or fake
                D_fake = self.D(fake_masks)
                # average critic output across batch
                D_fake = D_fake.mean(dim=0)
                D_fake.backward(one)
                D_fake_cost.update(D_fake.item(), self.batch_size(fake_masks))

                # train with gradient penalty
                gp = self._calc_gp(real_masks, fake_masks)
                gp.backward()

                D_cost = D_fake - D_real + gp
                Wasserstein_D = D_real - D_fake

                # log D_cost and Wasserstein_dist only for samples from the source domain
                n_batch = 2 * self.batch_size(fake_masks)
                D_losses.update(D_cost.item(), n_batch)
                Wasserstein_dist.update(Wasserstein_D.item(), n_batch)

                # update D's weights
                self.D_optimizer.step()

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
                self.writer.add_scalar('D_real_cost', D_real_cost.avg, self.num_iterations)
                self.writer.add_scalar('D_fake_cost', D_fake_cost.avg, self.num_iterations)

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

    @staticmethod
    def _extract_real_masks(target, combine_masks, label_smoothing):
        def _add_noise(mask, sigma=0.05):
            gaussian_noise = torch.randn(mask.size()).to(mask.device) * sigma
            mask += gaussian_noise
            return mask

        real_masks = []
        for tar in target:
            rms = []
            for i in torch.unique(tar):
                if i == 0:
                    # ignore 0-label
                    continue

                inst_mask = (tar == i).float()
                if label_smoothing:
                    # add noise to instance mask to prevent discriminator from converging too quickly
                    inst_mask = _add_noise(inst_mask)
                    # clamp values
                    inst_mask.clamp_(0, 1)

                # add channel dim and save real masks
                rms.append(inst_mask.unsqueeze(0))

            if combine_masks and len(rms) > 0:
                real_mask = (tar > 0).float()
                real_mask = real_mask.unsqueeze(0)
                real_mask.clamp_(0, 1)

                real_masks.append(real_mask)
            else:
                real_masks.extend(rms)

        if len(real_masks) == 0:
            return None

        real_masks = torch.stack(real_masks).to(target.device)
        return real_masks
