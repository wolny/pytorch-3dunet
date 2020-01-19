import torch
from torch import nn

from pytorch3dunet.unet3d.utils import expand_as_one_hot


class ContrastiveLoss(nn.Module):
    """
    Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function'
    """

    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001):
        super(ContrastiveLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_cluster_means(self, input, target):
        embedding_dims = input.size()[1]
        # expand target: NxCxDxHxW -> NxCxExDxHxW
        # NxCx1xDxHxW
        target = target.unsqueeze(2)
        # save target's copy in order to compute the average embeddings later
        target_copy = target.clone()
        shape = list(target.size())
        shape[2] = embedding_dims
        target = target.expand(shape)

        # expand input: NxExDxHxW -> Nx1xExDxHxW
        input = input.unsqueeze(1)

        # sum embeddings in each instance (multiply first via broadcasting) output: NxCxEx1x1x1
        embeddings_per_instance = input * target
        num = torch.sum(embeddings_per_instance, dim=(3, 4, 5), keepdim=True)
        # get number of voxels in each cluster output: NxCx1x1x1x1
        num_voxels_per_instance = torch.sum(target_copy, dim=(3, 4, 5), keepdim=True)
        # compute mean embeddings NxCxEx1x1x1
        mean_embeddings = num / num_voxels_per_instance
        # return mean embeddings and additional tensors needed for further computations
        return mean_embeddings, embeddings_per_instance

    def _compute_variance_term(self, cluster_means, embeddings_per_instance, target):
        # compute the distance to cluster means, result:(NxCxDxHxW)
        embedding_norms = torch.norm(embeddings_per_instance - cluster_means, self.norm, dim=2)
        # get per instance distances (apply instance mask)
        embedding_norms = embedding_norms * target
        # zero out distances less than delta_var and sum to get the variance (NxC)
        embedding_variance = torch.clamp(embedding_norms - self.delta_var, min=0) ** 2
        embedding_variance = torch.sum(embedding_variance, dim=(2, 3, 4))
        # get number of voxels per instance (NxC)
        num_voxels_per_instance = torch.sum(target, dim=(2, 3, 4))
        # normalize the variance term
        C = target.size()[1]
        variance_term = torch.sum(embedding_variance / num_voxels_per_instance, dim=1) / C
        return variance_term

    def _compute_distance_term(self, cluster_means, C):
        if C == 1:
            # just one cluster in the batch, so distance term does not contribute to the loss
            return 0.
        # squeeze space dims
        for _ in range(3):
            cluster_means = cluster_means.squeeze(-1)
        # expand cluster_means tensor in order to compute the pair-wise distance between cluster means
        cluster_means = cluster_means.unsqueeze(1)
        shape = list(cluster_means.size())
        shape[1] = C
        # NxCxCxEx1x1x1
        cm_matrix1 = cluster_means.expand(shape)
        # transpose the cluster_means matrix in order to compute pair-wise distances
        cm_matrix2 = cm_matrix1.permute(0, 2, 1, 3)
        # compute pair-wise distances (NxCxC)
        dist_matrix = torch.norm(cm_matrix1 - cm_matrix2, p=self.norm, dim=3)
        # create matrix for the repulsion distance (i.e. cluster centers further apart than 2 * delta_dist
        # are not longer repulsed)
        repulsion_dist = 2 * self.delta_dist * (1 - torch.eye(C))
        # 1xCxC
        repulsion_dist = repulsion_dist.unsqueeze(0).to(cluster_means.device)
        # zero out distances grater than 2*delta_dist (NxCxC)
        hinged_dist = torch.clamp(repulsion_dist - dist_matrix, min=0) ** 2
        # sum all of the hinged pair-wise distances
        hinged_dist = torch.sum(hinged_dist, dim=(1, 2))
        # normalized by the number of paris and return
        return hinged_dist / (C * (C - 1))

    def _compute_regularizer_term(self, cluster_means, C):
        # squeeze space dims
        for _ in range(3):
            cluster_means = cluster_means.squeeze(-1)
        norms = torch.norm(cluster_means, p=self.norm, dim=2)
        assert norms.size()[1] == C
        # return the average norm per batch
        return torch.sum(norms, dim=1).div(C)

    def forward(self, input, target):
        """
        Args:
             input (torch.tensor): embeddings predicted by the network (NxExDxHxW) (E - embedding dims)
             target (torch.tensor): ground truth instance segmentation (NxDxHxW)

        Returns:
            Combined loss defined as: alpha * variance_term + beta * distance_term + gamma * regularization_term
        """
        # get number of instances in the batch
        C = torch.unique(target).size()[0]
        # expand each label as a one-hot vector: N x D x H x W -> N x C x D x H x W
        target = expand_as_one_hot(target, C)
        # compare spatial dimensions
        assert input.dim() == target.dim() == 5
        assert input.size()[2:] == target.size()[2:]

        # compute mean embeddings and assign embeddings to instances
        cluster_means, embeddings_per_instance = self._compute_cluster_means(input, target)
        variance_term = self._compute_variance_term(cluster_means, embeddings_per_instance, target)
        distance_term = self._compute_distance_term(cluster_means, C)
        regularization_term = self._compute_regularizer_term(cluster_means, C)
        # total loss
        loss = self.alpha * variance_term + self.beta * distance_term + self.gamma * regularization_term
        # reduce batch dimension
        return torch.mean(loss)
