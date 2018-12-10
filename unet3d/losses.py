import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and then simply taking the average.
    Since it's not a loss function, no need to compute gradients and thus no need to subclass nn.Module.
    """

    def __init__(self, should_normalize=False, epsilon=1e-5):
        if should_normalize:
            self.normalizer = nn.Softmax()
        else:
            self.normalizer = None
        self.epsilon = epsilon

    def __call__(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        if self.normalizer is not None:
            input = self.normalizer(input)

        input = flatten(input)
        target = flatten(target)

        # Compute per channel Dice Coefficient
        intersect = (input * target).sum(-1) + self.epsilon
        denominator = (input + target).sum(-1) + self.epsilon
        # Average across channels in order to get the final score
        return torch.mean(2. * intersect / denominator)


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)

        target_sum = target.sum(-1)
        class_weights = 1. / (target_sum * target_sum + self.epsilon)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum() + self.epsilon

        denominator = ((input + target).sum(-1) * class_weights).sum() + self.epsilon
        return 1 - 2. * intersect / denominator


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        class_weights = self._class_weights(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, _stacklevel=5)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class IgnoreIndexLossWrapper:
    """
    Wrapper around loss functions which do not support 'ignore_index', e.g. BCELoss.
    Throws exception if the wrapped loss supports the 'ignore_index' option.
    """

    def __init__(self, loss_criterion, ignore_index=-1):
        if hasattr(loss_criterion, 'ignore_index'):
            raise RuntimeError(f"Cannot wrap {type(loss_criterion)}. 'Use ignore_index' attribute instead")
        self.loss_criterion = loss_criterion
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
        masked_input = input * mask
        masked_target = target * mask
        return self.loss_criterion(masked_input, masked_target)
