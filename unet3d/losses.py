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


class DiceCoefficient(nn.Module):
    """Computes Dice Coefficient
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and then simply taking the average.
    """

    def __init__(self, epsilon=1e-5):
        super(DiceCoefficient, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)

        # Compute per channel Dice Coefficient
        intersect = (input * target).sum(-1) + self.epsilon
        denominator = (input + target).sum(-1) + self.epsilon
        # Average across channels in order to get the final score
        return torch.mean(2. * intersect / denominator)


class GeneralizedDiceLoss(DiceCoefficient):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5):
        super(GeneralizedDiceLoss, self).__init__(epsilon)

    def forward(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)

        target_sum = target.sum(-1)
        class_weights = 1. / (target_sum * target_sum + self.epsilon)

        # Compute per channel Dice Coefficient
        intersect = ((input * target).sum(-1) * class_weights).sum() + self.epsilon
        denominator = ((input + target).sum(-1) * class_weights).sum() + self.epsilon
        # Average across channels in order to get the final score
        return 1 - 2. * intersect / denominator


class WeightedNLLLoss(nn.Module):
    def __init__(self):
        super(WeightedNLLLoss, self).__init__()

    def forward(self, input, target):
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return F.nll_loss(input, target, weight=class_weights)
