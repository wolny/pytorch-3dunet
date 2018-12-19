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

    def __init__(self, epsilon=1e-5, ignore_index=None):
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        # input and target shapes must match
        if target.dim() == 4:
            target = expand_target(target, C=input.size()[1], ignore_index=self.ignore_index)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        # Compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        denominator = (input + target).sum(-1) + self.epsilon
        # Average across channels in order to get the final score
        return torch.mean(2. * intersect / denominator)


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # input and target shapes must match
        if target.dim() == 4:
            target = expand_target(target, C=input.size()[1], ignore_index=self.ignore_index)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target_sum = target.sum(-1)
        class_weights = 1. / (target_sum * target_sum + self.epsilon)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum() + self.epsilon
        return 1 - 2. * intersect / denominator


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

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
            raise RuntimeError(f"Cannot wrap {type(loss_criterion)}. Use 'ignore_index' attribute instead")
        self.loss_criterion = loss_criterion
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        # always expand target tensor, so that input.size() == target.size()
        if target.dim() == 4:
            target = expand_target(target, C=input.size()[1], ignore_index=self.ignore_index)

        assert input.size() == target.size()

        mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
        masked_input = input * mask
        masked_target = target * mask
        return self.loss_criterion(masked_input, masked_target)


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax()

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_target(target, C=input.size()[1], ignore_index=self.ignore_index)
        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
            log_probabilities = log_probabilities * mask
            target = target * mask
        # compute the losses (this works cause the weights (NxDxHxW) will be brodcasted onto targets (NxCxDxHxW))
        result = -weights * target * log_probabilities
        # apply class_weights if present
        if self.class_weights is not None:
            result = flatten(result)
            class_weights = Variable(self.class_weights.unsqueeze(1), requires_grad=False)
            assert result.size()[0] == class_weights.size()[0]
        # average the losses
        return result.mean()


def expand_target(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4
    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)

    result = torch.zeros(shape)
    # for each batch instance
    for i in range(input.size()[0]):
        # iterate over channel axis and create corresponding binary mask in the target
        for c in range(C):
            mask = result[i, c]
            mask[input[i] == c] = 1
            if ignore_index is not None:
                mask[input[i] == ignore_index] = ignore_index
    return result.to(input.device)
