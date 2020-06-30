import torch

from pytorch3dunet.unet3d.utils import expand_as_one_hot


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
