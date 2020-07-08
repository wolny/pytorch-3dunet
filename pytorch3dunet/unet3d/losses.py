import math
import os

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss, BCELoss

from pytorch3dunet.unet3d.model import get_model, WGANDiscriminator
from pytorch3dunet.unet3d.utils import expand_as_one_hot, load_checkpoint, get_logger

logger = get_logger('Losses')


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class TagsAngularLoss(nn.Module):
    def __init__(self, tags_coefficients):
        super(TagsAngularLoss, self).__init__()
        self.tags_coefficients = tags_coefficients

    def forward(self, inputs, targets, weight):
        assert isinstance(inputs, list)
        # if there is just one output head the 'inputs' is going to be a singleton list [tensor]
        # and 'targets' is just going to be a tensor (that's how the HDF5Dataloader works)
        # so wrap targets in a list in this case
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets) == len(self.tags_coefficients)
        loss = 0
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients):
            loss += alpha * square_angular_loss(input, target, weight)

        return loss


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


def square_angular_loss(input, target, weights=None):
    """
    Computes square angular loss between input and target directions.
    Makes sure that the input and target directions are normalized so that torch.acos would not produce NaNs.

    :param input: 5D input tensor (NCDHW)
    :param target: 5D target tensor (NCDHW)
    :param weights: 3D weight tensor in order to balance different instance sizes
    :return: per pixel weighted sum of squared angular losses
    """
    assert input.size() == target.size()
    # normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
    stability_coeff = 0.999999
    input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    # compute cosine map
    cosines = (input * target).sum(dim=1)
    error_radians = torch.acos(cosines)
    if weights is not None:
        return (error_radians * error_radians * weights).sum()
    else:
        return (error_radians * error_radians).sum()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config['device'])

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        # convert to cuda tensor if necessary
        pos_weight = torch.tensor(pos_weight).to(config['device'])

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    return loss


################################################# embedding losses ####################################################
def check_consecutive(labels):
    """ Check that the input labels are consecutive and start at zero.
    """
    diff = labels[1:] - labels[:-1]
    return (labels[0] == 0) and (diff == 1).all()


class _AbstractContrastiveLoss(nn.Module):
    """
    Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function'

    This implementation expands all tensors to match the instance dimensions.
    This means that it's fast, but has high memory consumption.
    Also, the implementation does not support masking any instance labels in the loss.
    """

    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001, delta=1.,
                 ignore_zero_in_variance=False, ignore_zero_in_distance=False, aux_loss_ignore_zero=True):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.ignore_zero_in_variance = ignore_zero_in_variance
        self.ignore_zero_in_distance = ignore_zero_in_distance
        self.aux_loss_ignore_zero = aux_loss_ignore_zero

    def _compute_cluster_means(self, input_, target, spatial_ndim):
        """
        Computes mean embeddings per instance, embeddings withing a given instance and the number of voxels per instance.

        C - number of instances
        E - embedding dimension
        SPATIAL - volume shape, i.e. DxHxW for 3D/ HxW for 2D

        Args:
            input_: tensor of pixel embeddings, shape: ExSPATIAL
            target: one-hot encoded target instances, shape: CxSPATIAL
            spatial_ndim: rank of the spacial tensor
        """
        dim_arg = (2, 3) if spatial_ndim == 2 else (2, 3, 4)

        embedding_dim = input_.size()[0]

        # get number of voxels in each cluster output: Cx1x1(SPATIAL)
        num_voxels_per_instance = torch.sum(target, dim=dim_arg, keepdim=True)

        # expand target: Cx1xSPATIAL -> CxExSPATIAL
        shape = list(target.size())
        shape[1] = embedding_dim
        target = target.expand(shape)

        # expand input_: ExSPATIAL -> 1xExSPATIAL
        input_ = input_.unsqueeze(0)

        # sum embeddings in each instance (multiply first via broadcasting); embeddings_per_instance shape CxExSPATIAL
        embeddings_per_instance = input_ * target
        # num's shape: CxEx1(SPATIAL)
        num = torch.sum(embeddings_per_instance, dim=dim_arg, keepdim=True)

        # compute mean embeddings per instance CxEx1(SPATIAL)
        mean_embeddings = num / num_voxels_per_instance

        # return mean embeddings and additional tensors needed for further computations
        return mean_embeddings, embeddings_per_instance, num_voxels_per_instance

    def _compute_variance_term(self, cluster_means, embeddings_per_instance, target, num_voxels_per_instance, C,
                               spatial_ndim, ignore_zero_label):
        """
        Computes the variance term, i.e. intra-cluster pull force that draws embeddings towards the mean embedding

        C - number of clusters (instances)
        E - embedding dimension
        SPATIAL - volume shape, i.e. DxHxW for 3D/ HxW for 2D
        SPATIAL_SINGLETON - singleton dim with the rank of the volume, i.e. (1,1,1) for 3D, (1,1) for 2D

        Args:
            cluster_means: mean embedding of each instance, tensor (CxExSPATIAL_SINGLETON)
            embeddings_per_instance: embeddings vectors per instance, tensor (CxExSPATIAL); for a given instance `k`
                embeddings_per_instance[k, ...] contains 0 outside of the instance mask target[k, ...]
            target: instance mask, tensor (CxSPATIAL); each label is represented as one-hot vector
            num_voxels_per_instance: number of voxels per instance Cx1x1(SPATIAL)
            C: number of instances (clusters)
            spatial_ndim: rank of the spacial tensor
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """

        dim_arg = (2, 3) if spatial_ndim == 2 else (2, 3, 4)

        # compute the distance to cluster means, (norm across embedding dimension); result:(Cx1xSPATIAL)
        dist_to_mean = torch.norm(embeddings_per_instance - cluster_means, self.norm, dim=1, keepdim=True)

        # get distances to mean embedding per instance (apply instance mask)
        dist_to_mean = dist_to_mean * target

        if ignore_zero_label:
            # zero out distances corresponding to 0-label cluster, so that it does not contribute to the loss
            dist_mask = torch.ones_like(dist_to_mean)
            dist_mask[0] = 0
            dist_to_mean = dist_to_mean * dist_mask
            # decrease number of instances
            C -= 1
            # if there is only 0-label in the target return 0
            if C == 0:
                return 0.

        # zero out distances less than delta_var (hinge)
        hinge_dist = torch.clamp(dist_to_mean - self.delta_var, min=0) ** 2
        # sum up hinged distances
        dist_sum = torch.sum(hinge_dist, dim=dim_arg, keepdim=True)

        # normalize the variance term
        variance_term = torch.sum(dist_sum / num_voxels_per_instance) / C
        return variance_term

    def _compute_distance_term(self, cluster_means, C, ignore_zero_label):
        """
        Compute the distance term, i.e an inter-cluster push-force that pushes clusters away from each other, increasing
        the distance between cluster centers

        Args:
            cluster_means: mean embedding of each instance, tensor (CxExSPATIAL_SINGLETON)
            C: number of instances (clusters)
            spatial_ndim: rank of the spacial tensor
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """
        if C == 1:
            # just one cluster in the batch, so distance term does not contribute to the loss
            return 0.

        # expand cluster_means tensor in order to compute the pair-wise distance between cluster means
        # CxE -> CxCxE
        cluster_means = cluster_means.unsqueeze(1)
        shape = list(cluster_means.size())
        shape[1] = C

        # cm_matrix1 is CxCxE
        cm_matrix1 = cluster_means.expand(shape)
        # transpose the cluster_means matrix in order to compute pair-wise distances
        cm_matrix2 = cm_matrix1.permute(1, 0, 2)
        # compute pair-wise distances between cluster means, result is a CxC tensor
        dist_matrix = torch.norm(cm_matrix1 - cm_matrix2, p=self.norm, dim=2)

        # create matrix for the repulsion distance (i.e. cluster centers further apart than 2 * delta_dist
        # are not longer repulsed)
        repulsion_dist = 2 * self.delta_dist * (1 - torch.eye(C))
        repulsion_dist = repulsion_dist.to(cluster_means.device)

        if ignore_zero_label:
            if C == 2:
                # just two cluster instances, including one which is ignored, i.e. distance term does not contribute to the loss
                return 0.
            # set the distance to 0-label to be 2*delta_dist + epsilon so that it does not contribute to the loss because of the hinge at 2*delta_dist
            dist_mask = torch.ones_like(dist_matrix)
            ignore_dist = 2 * self.delta_dist + 1e-4
            dist_mask[0, 1:] = ignore_dist
            dist_mask[1:, 0] = ignore_dist
            # mask the dist_matrix
            dist_matrix = dist_matrix * dist_mask
            # decrease number of instances
            C -= 1

        # zero out distances grater than 2*delta_dist (hinge)
        hinged_dist = torch.clamp(repulsion_dist - dist_matrix, min=0) ** 2
        # sum all of the hinged pair-wise distances
        dist_sum = torch.sum(hinged_dist)
        # normalized by the number of paris and return
        distance_term = dist_sum / (C * (C - 1))
        return distance_term

    def _compute_regularizer_term(self, cluster_means, C):
        """
        Computes the regularizer term, i.e. a small pull-force that draws all clusters towards origin to keep
        the network activations bounded
        """
        # compute the norm of the mean embeddings
        norms = torch.norm(cluster_means, p=self.norm, dim=1)
        assert norms.size()[0] == C
        # return the average norm per batch
        regularizer_term = torch.sum(norms) / C
        return regularizer_term

    def auxiliary_loss(self, embeddings, cluster_means, target):
        """
        Computes auxiliary loss based on embeddings and a given list of target instances together with their mean embeddings

        Args:
            embeddings (torch.tensor): pixel embeddings (ExSPATIAL)
            cluster_means (torch.tensor): mean embeddings per instance (CxExSINGLETON_SPATIAL)
            target (torch.tensor): ground truth instance segmentation (SPATIAL)
        """
        raise NotImplementedError

    def forward(self, input_, target):
        """
        Args:
             input_ (torch.tensor): embeddings predicted by the network (NxExDxHxW) (E - embedding dims)
                                    expects float32 tensor
             target (torch.tensor): ground truth instance segmentation (NxDxHxW)
                                    expects int64 tensor
                                    if self.ignore_zero_label is True then expects target of shape Nx2xDxHxW where
                                    relabeled version is in target[:,0,...] and the original labeling is in target[:,1,...]

        Returns:
            Combined loss defined as: alpha * variance_term + beta * distance_term + gamma * regularization_term
        """

        n_batches = input_.shape[0]
        # compute the loss per each instance in the batch separately
        # and sum it up in the per_instance variable
        per_instance_loss = 0.
        for single_input, single_target in zip(input_, target):
            ignore_zero_in_variance, ignore_zero_in_distance, single_target = self._should_ignore_zero(single_target)
            # save original target tensor
            orig_target = single_target

            # get number of instances in the batch instance
            instances = torch.unique(single_target)
            assert check_consecutive(instances)
            # get the number of instances
            C = instances.size()[0]

            # SPATIAL = D X H X W in 3d case, H X W in 2d case
            # expand each label as a one-hot vector: SPATIAL -> C x SPATIAL
            # `expand_as_one_hot` requires batch dimension; later so we need to squeeze the result
            single_target = expand_as_one_hot(single_target.unsqueeze(0), C).squeeze(0)

            # compare shapes of input and output; single_input is ExSPATIAL, single_target is CxSPATIAL
            assert single_input.dim() in (3, 4)
            assert single_input.dim() == single_target.dim()
            # compare spatial dimensions
            assert single_input.size()[1:] == single_target.size()[1:]
            spatial_dims = single_input.dim() - 1

            # expand target: CxSPATIAL -> Cx1xSPATIAL for further computation
            single_target = single_target.unsqueeze(1)
            # compute mean embeddings, assign embeddings to instances and get the number of voxels per instance
            cluster_means, embeddings_per_instance, num_voxels_per_instance = self._compute_cluster_means(single_input,
                                                                                                          single_target,
                                                                                                          spatial_dims)

            # compute variance term, i.e. pull force
            variance_term = self._compute_variance_term(cluster_means, embeddings_per_instance,
                                                        single_target, num_voxels_per_instance,
                                                        C, spatial_dims, ignore_zero_in_variance)

            # compute the auxiliary loss
            aux_loss = self.auxiliary_loss(single_input, cluster_means, orig_target)

            # squeeze spatial dims
            for _ in range(spatial_dims):
                cluster_means = cluster_means.squeeze(-1)

            # compute distance term, i.e. push force
            distance_term = self._compute_distance_term(cluster_means, C, ignore_zero_in_distance)

            # compute regularization term
            # do not ignore 0-label in the regularizer, we still want the activations of 0-label to be bounded
            regularization_term = self._compute_regularizer_term(cluster_means, C)

            # compute total loss and sum it up
            loss = self.alpha * variance_term + \
                   self.beta * distance_term + \
                   self.gamma * regularization_term + \
                   self.delta * aux_loss

            per_instance_loss += loss

        # reduce across the batch dimension
        return per_instance_loss.div(n_batches)

    def _should_ignore_zero(self, target):
        # set default values
        ignore_zero_in_variance = False
        ignore_zero_in_distance = False
        single_target = target

        if self.ignore_zero_in_variance or self.ignore_zero_in_distance:
            assert target.dim() == 4, "Expects target to be 2xDxHxW when ignore_zero_label is True"
            # get relabeled target
            single_target = target[0]
            # get original target and ignore 0-label only if 0-label was present in the original target
            original = target[1]
            ignore_zero_in_variance = self.ignore_zero_in_variance and (0 in original)
            ignore_zero_in_distance = self.ignore_zero_in_distance and (0 in original)

        return ignore_zero_in_variance, ignore_zero_in_distance, single_target


class ContrastiveLoss(_AbstractContrastiveLoss):
    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001,
                 ignore_zero_in_variance=False, ignore_zero_in_distance=False):
        super(ContrastiveLoss, self).__init__(delta_var, delta_dist, norm=norm,
                                              alpha=alpha, beta=beta, gamma=gamma, delta=0.,
                                              ignore_zero_in_variance=ignore_zero_in_variance,
                                              ignore_zero_in_distance=ignore_zero_in_distance)

    def auxiliary_loss(self, embeddings, cluster_means, target):
        # no auxiliary loss in the standard ContrastiveLoss
        return 0.


class LovaszSoftmaxLoss(nn.Module):
    """
    Copied from: https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
    """

    def forward(self, input, target):
        return self.lovasz_hinge_flat(*self.flatten_binary_scores(input, target, ignore=None))

    @staticmethod
    def lovasz_hinge_flat(logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = LovaszSoftmaxLoss.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    @staticmethod
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    @staticmethod
    def flatten_binary_scores(scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels


class GANShapePriorLoss(nn.Module):
    def __init__(self, model_path, D_model_config):
        super().__init__()
        assert model_path is not None
        assert D_model_config is not None

        # load model
        D = get_model(D_model_config)
        # send to device
        D = D.to(torch.device(D_model_config['device']))
        D.eval()

        load_checkpoint(model_path, D, model_key='D_model_state_dict')
        # freeze weights
        for p in D.parameters():
            p.requires_grad = False

        if isinstance(D, WGANDiscriminator):
            self.loss = self.WGANLoss(D)
        else:
            self.loss = self.GANLoss(D)

    def forward(self, inst_pmap, inst_mask):
        # add batch and channel dimensions
        inst_pmap = inst_pmap.view((1, 1) + inst_pmap.size())
        return self.loss(inst_pmap)

    class WGANLoss:
        def __init__(self, D):
            self.D = D

        def __call__(self, inst_pmap):
            return -self.D(inst_pmap)

    class GANLoss:
        def __init__(self, D):
            self.D = D
            self.bce_loss = nn.BCELoss()

        def __call__(self, inst_pmap):
            real_labels = torch.ones(inst_pmap.size(0), 1).to(inst_pmap.device)
            outputs = self.D(inst_pmap)
            return self.bce_loss(outputs, real_labels)


class AuxContrastiveLoss(_AbstractContrastiveLoss):
    def __init__(self, delta_var, delta_dist, aux_loss, dist_to_mask_conf,
                 norm='fro', alpha=1., beta=1., gamma=0.001, delta=1.,
                 aux_loss_ignore_zero=True, model_path=None, D_model_config=None,
                 log_aux_after=250, checkpoint_dir=None):
        super().__init__(delta_var, delta_dist, norm=norm, alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                         ignore_zero_in_variance=False,
                         ignore_zero_in_distance=False,
                         aux_loss_ignore_zero=aux_loss_ignore_zero)
        # ignore instance corresponding to 0-label
        self.delta_var = delta_var
        # init auxiliary loss
        assert aux_loss in ['bce', 'dice', 'lovasz', 'gan']
        if aux_loss == 'bce':
            self.aux_loss = BCELoss()
        elif aux_loss == 'dice':
            self.aux_loss = DiceLoss(normalization='none')
        elif aux_loss == 'lovasz':
            self.aux_loss = LovaszSoftmaxLoss()
        else:
            self.aux_loss = GANShapePriorLoss(model_path=model_path, D_model_config=D_model_config)

        # init dist_to_mask function which maps per-instance distance map to the instance probability map
        self.dist_to_mask = self._create_dist_to_mask_fun(dist_to_mask_conf, delta_var)
        self.log_aux_after = log_aux_after
        self.aux_invocations = 0
        if checkpoint_dir is not None:
            self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        else:
            self.writer = None

    def _create_dist_to_mask_fun(self, dist_to_mask_conf, delta_var):
        name = dist_to_mask_conf['name']
        assert name in ['logistic', 'gaussian']
        if name == 'logistic':
            return self.Logistic(delta_var, dist_to_mask_conf.get('k', 10.))
        else:
            return self.Gaussian(delta_var, dist_to_mask_conf.get('pmaps_threshold', 0.5))

    def auxiliary_loss(self, embeddings, cluster_means, target):
        self.aux_invocations += 1

        assert embeddings.size()[1:] == target.size()

        per_instance_loss = 0.
        for i, cm in enumerate(cluster_means):
            if i == 0 and self.aux_loss_ignore_zero:
                # ignore 0-label
                continue
            # compute distance map; embeddings is ExSPATIAL, cluster_mean is ExSINGLETON_SPATIAL, so we can just broadcast
            dist_to_mean = torch.norm(embeddings - cm, self.norm, dim=0)
            # convert distance map to instance pmaps
            inst_pmap = self.dist_to_mask(dist_to_mean)
            # compute the auxiliary loss between the instance_pmap and the ground truth instance mask
            assert i in target
            inst_mask = (target == i).float()
            loss = self.aux_loss(inst_pmap, inst_mask)
            per_instance_loss += loss.sum()

        if self.aux_invocations % self.log_aux_after == 0 and self.writer is not None:
            aux_loss = per_instance_loss.data.cpu().numpy()
            aux_loss_grad = per_instance_loss.grad.data.cpu().numpy()
            logger.info(f'Aux Loss: {aux_loss}, Aux Loss Grad: {aux_loss_grad}')
            self.writer.add_scalar('aux_loss', aux_loss, self.aux_invocations)
            self.writer.add_scalar('aux_loss_grad', aux_loss_grad, self.aux_invocations)

        return per_instance_loss

    # below are the kernel function used to convert the distance map (i.e. `||embeddings - anchor_embedding||`)
    # into an instance mask
    class Logistic(nn.Module):
        def __init__(self, delta_var, k=10.):
            super().__init__()
            self.delta_var = delta_var
            self.k = k

        def forward(self, dist_map):
            return torch.sigmoid(-self.k * (dist_map - self.delta_var))

    class Gaussian(nn.Module):
        def __init__(self, delta_var, pmaps_threshold):
            super().__init__()
            # dist_var^2 = -2*sigma*ln(pmaps_threshold)
            self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

        def forward(self, dist_map):
            return torch.exp(- dist_map * dist_map / self.two_sigma)


class SegEmbLoss(nn.Module):
    def __init__(self, delta_var, delta_dist, w1=1., w2=1.):
        super(SegEmbLoss, self).__init__()
        self.w1 = w1
        self.bce_dice_loss = BCEDiceLoss(alpha=1., beta=1.)
        self.w2 = w2
        self.contrastive_loss = ContrastiveLoss(delta_var=delta_var, delta_dist=delta_dist)

    def forward(self, input, target):
        assert isinstance(input, tuple)
        seg_pmaps, embeddings = input

        seg_target = target[:, :1, ...]
        inst_target = target[:, -1, ...]

        return self.w1 * self.bce_dice_loss(seg_pmaps, seg_target.float()) + \
               self.w2 * self.contrastive_loss(embeddings, inst_target)


#######################################################################################################################

def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alphs', 1.)
        beta = loss_config.get('beta', 1.)
        return BCEDiceLoss(alpha, beta)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return GeneralizedDiceLoss(normalization=normalization)
    elif name == 'DiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return DiceLoss(weight=weight, normalization=normalization)
    elif name == 'TagsAngularLoss':
        tags_coefficients = loss_config['tags_coefficients']
        return TagsAngularLoss(tags_coefficients)
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'ContrastiveLoss':
        return ContrastiveLoss(loss_config['delta_var'],
                               loss_config['delta_dist'],
                               loss_config['norm'],
                               loss_config['alpha'],
                               loss_config['beta'],
                               loss_config['gamma'],
                               loss_config.get('ignore_zero_in_variance', False),
                               loss_config.get('ignore_zero_in_distance', False))
    elif name == 'AuxContrastiveLoss':
        return AuxContrastiveLoss(loss_config['delta_var'],
                                  loss_config['delta_dist'],
                                  loss_config['aux_loss'],
                                  loss_config['dist_to_mask'],
                                  loss_config['norm'],
                                  loss_config['alpha'],
                                  loss_config['beta'],
                                  loss_config['gamma'],
                                  loss_config['delta'],
                                  loss_config.get('aux_loss_ignore_zero', True),
                                  loss_config.get('model_path', None),
                                  loss_config.get('D_model', None),
                                  loss_config.get('log_aux_after', None),
                                  loss_config.get('checkpoint_dir', None))
    elif name == 'SegEmbLoss':
        return SegEmbLoss(loss_config['delta_var'],
                          loss_config['delta_dist'],
                          loss_config.get('w1', 1.),
                          loss_config.get('w2', 1.))
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss(threshold=loss_config['threshold'], initial_weight=loss_config['initial_weight'],
                                    apply_below_threshold=loss_config.get('apply_below_threshold', True))
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
