import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.losses import get_loss_criterion, _AbstractContrastiveLoss, AuxContrastiveLoss
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import expand_as_one_hot, get_number_of_learnable_parameters, get_logger, \
    create_optimizer, create_lr_scheduler, get_tensorboard_formatter, create_sample_plotter, load_checkpoint, \
    save_checkpoint, RunningAverage


class MeanEmbeddingAnchor:
    def __init__(self, c_mean_fn):
        """
        Extracts mean instance embedding as an anchor embedding.

        :param c_mean_fn: function to extract mean embeddings given the embeddings and target masks
        """
        self.c_mean_fn = c_mean_fn

    def __call__(self, emb, tar):
        instances = torch.unique(tar)
        C = instances.size(0)

        single_target = expand_as_one_hot(tar.unsqueeze(0), C).squeeze(0)
        single_target = single_target.unsqueeze(1)
        spatial_dims = emb.dim() - 1

        cluster_means, _, _ = self.c_mean_fn(emb, single_target, spatial_dims)
        return cluster_means


class RandomEmbeddingAnchor:
    """
    Selects a random pixel inside an instance, gets its embedding and uses is as an anchor embedding
    """

    def __call__(self, emb, tar):
        instances = torch.unique(tar)
        anchor_embeddings = []
        for i in instances:
            z, y, x = torch.nonzero(tar == i, as_tuple=True)
            ind = torch.randint(len(z), (1,))[0]
            anchor_emb = emb[:, z[ind], y[ind], x[ind]]
            anchor_embeddings.append(anchor_emb)

        result = torch.stack(anchor_embeddings, dim=0).to(emb.device)
        # expand dimensions
        result = result[..., None, None, None]
        return result


def extract_instance_masks(embeddings, target, anchor_embeddings_extractor, dist_to_mask_fn, combine_masks,
                           label_smoothing=True):
    """
    Extract instance masks given the embeddings, target,
    anchor embeddings extraction functor (anchor_embeddings_extractor),
    kernel function computing distance to anchor (dist_to_mask_fn)
    and whether to combine the masks or not (combine_masks)
    """

    def _add_noise(mask, sigma=0.05):
        gaussian_noise = torch.randn(mask.size()).to(mask.device) * sigma
        mask += gaussian_noise
        return mask

    # iterate over batch
    real_masks = []
    fake_masks = []

    for emb, tar in zip(embeddings, target):
        anchor_embeddings = anchor_embeddings_extractor(emb, tar)
        rms = []
        fms = []
        for i, anchor_emb in enumerate(anchor_embeddings):
            if i == 0:
                # ignore 0-label
                continue

            # compute distance map; embeddings is ExSPATIAL, anchor_embeddings is ExSINGLETON_SPATIAL, so we can just broadcast
            dist_to_mean = torch.norm(emb - anchor_emb, 'fro', dim=0)
            # convert distance map to instance pmaps
            inst_pmap = dist_to_mask_fn(dist_to_mean)
            # add channel dim and save fake masks
            fms.append(inst_pmap.unsqueeze(0))

            assert i in target

            inst_mask = (tar == i).float()
            if label_smoothing:
                # add noise to instance mask to prevent discriminator from converging too quickly
                inst_mask = _add_noise(inst_mask)
                # clamp values
                inst_mask.clamp_(0, 1)

            # add channel dim and save real masks
            rms.append(inst_mask.unsqueeze(0))

        if combine_masks and len(fms) > 0:
            fake_mask = torch.zeros_like(fms[0])
            for fm in fms:
                fake_mask += fm

            real_mask = (tar > 0).float()
            real_mask = real_mask.unsqueeze(0)
            real_mask = _add_noise(real_mask)
            real_mask.clamp_(0, 1)

            real_masks.append(real_mask)
            fake_masks.append(fake_mask)
        else:
            real_masks.extend(rms)
            fake_masks.extend(fms)

    if len(real_masks) == 0:
        return None, None

    real_masks = torch.stack(real_masks).to(embeddings.device)
    fake_masks = torch.stack(fake_masks).to(embeddings.device)
    return real_masks, fake_masks


logger = get_logger('AbstractGANTrainer')


class AbstractEmbeddingGANTrainerBuilder:

    @staticmethod
    def trainer_class():
        raise NotImplementedError

    @classmethod
    def build(cls, config):
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
            return cls.trainer_class().from_checkpoint(
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
            return cls.trainer_class().from_pretrained_emb(
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
            return cls.trainer_class()(
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


class AbstractEmbeddingGANTrainer:
    def __init__(self, G, D,
                 G_optimizer, D_optimizer,
                 G_lr_scheduler,
                 G_loss_criterion, G_eval_criterion,
                 gan_loss_weight,
                 device, loaders, checkpoint_dir,
                 combine_masks=False, anchor_extraction='mean', label_smoothing=True,
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
        self.G_iterations = 1
        self.D_iterations = 1
        self.log_after_iters = log_after_iters
        self.validate_after_iters = validate_after_iters
        self.max_num_iterations = max_num_iterations
        self.max_num_epochs = max_num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.loaders = loaders
        self.device = device
        self.G_eval_criterion = G_eval_criterion
        self.G_loss_criterion = G_loss_criterion
        self.gan_loss_weight = gan_loss_weight
        self.G_lr_scheduler = G_lr_scheduler
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.D = D
        self.G = G
        self.combine_masks = combine_masks
        assert anchor_extraction in ['mean', 'random']
        if anchor_extraction == 'mean':
            assert isinstance(self.G_loss_criterion, _AbstractContrastiveLoss)
            # function for computing a mean embeddings of target instances
            c_mean_fn = self.G_loss_criterion._compute_cluster_means
            self.anchor_embeddings_extractor = MeanEmbeddingAnchor(c_mean_fn)
        else:
            self.anchor_embeddings_extractor = RandomEmbeddingAnchor()
        self.label_smoothing = label_smoothing

        logger.info('GENERATOR')
        logger.info(self.G)
        logger.info('CRITIC/DISCRIMINATOR')
        logger.info(self.D)
        logger.info(f'gan_loss_weight: {gan_loss_weight}')

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

    @classmethod
    def from_pretrained_emb(cls, pre_trained,
                            G, D, G_optimizer, D_optimizer, G_lr_scheduler,
                            G_loss_criterion, G_eval_criterion,
                            loaders, tensorboard_formatter, sample_plotter, **kwargs):
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

    @classmethod
    def from_checkpoint(cls, checkpoint_path,
                        G, D,
                        G_optimizer, D_optimizer, G_lr_scheduler,
                        G_loss_criterion, G_eval_criterion,
                        loaders,
                        tensorboard_formatter=None, sample_plotter=None,
                        **kwargs):
        raise NotImplementedError

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
        raise NotImplementedError

    def validate(self):
        logger.info('Validating...')

        val_losses = RunningAverage()
        val_scores = RunningAverage()

        if self.sample_plotter is not None:
            self.sample_plotter.update_current_dir()

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, target = self.split_training_batch(t)

                output = self.G(input)
                loss = self.G_loss_criterion(output, target)
                val_losses.update(loss.item(), self.batch_size(input))

                eval_score = self.G_eval_criterion(output, target)
                val_scores.update(eval_score.item(), self.batch_size(input))

                if self.sample_plotter is not None:
                    self.sample_plotter(i, input, output, target, 'val')

            self.writer.add_scalar('val_embedding_loss', val_losses.avg, self.num_iterations)
            self.writer.add_scalar('val_eval', val_scores.avg, self.num_iterations)
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg

    def split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        assert len(t) == 2, f"Expected tuple of size 2 (input, target), but {len(t)} was given"
        input, target = _move_to_device(t)
        return input, target

    def is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

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

    @staticmethod
    def batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    def freeze_D(self):
        for p in self.D.parameters():
            p.requires_grad = False

    def unfreeze_D(self):
        for p in self.D.parameters():
            p.requires_grad = True

    def log_G_lr(self):
        lr = self.G_optimizer.param_groups[0]['lr']
        self.writer.add_scalar('G_learning_rate', lr, self.num_iterations)

    def log_images(self, inputs_map):
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

    def log_params(self, model, model_name):
        for name, value in model.named_parameters():
            self.writer.add_histogram(model_name + '/' + name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(model_name + '/' + name + '/grad', value.grad.data.cpu().numpy(),
                                      self.num_iterations)

    def save_checkpoint(self, is_best):
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
