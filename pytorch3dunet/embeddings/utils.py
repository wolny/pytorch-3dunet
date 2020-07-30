import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import expand_as_one_hot, get_number_of_learnable_parameters, get_logger, \
    create_optimizer, create_lr_scheduler, get_tensorboard_formatter, create_sample_plotter


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
