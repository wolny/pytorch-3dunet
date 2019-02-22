import torch

from unet3d.losses import compute_per_channel_dice, expand_as_one_hot


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-5, ignore_index=None):
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. If the tensor is 4D (NxDxHxW) it will be expanded to 5D as one-hot
        :return: Soft Dice Coefficient averaged over all channels/classes
        """
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        n_classes = input.size()[1]
        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        # batch dim must be 1
        input = input[0]
        target = target[0]
        assert input.size() == target.size()

        binary_prediction = self._binarize_predictions(input)

        if self.ignore_index is not None:
            # zero out ignore_index
            mask = target == self.ignore_index
            binary_prediction[mask] = 0
            target[mask] = 0

        # convert to uint8 just in case
        binary_prediction = binary_prediction.byte()
        target = target.byte()

        per_channel_iou = []
        for c in range(n_classes):
            if c in self.skip_channels:
                continue

            per_channel_iou.append(self._jaccard_index(binary_prediction[c], target[c]))

        assert per_channel_iou, "All channels were ignored from the computation"
        return torch.mean(torch.tensor(per_channel_iou))

    def _binarize_predictions(self, input):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        :return:
        """
        return torch.sum(prediction & target).float() / torch.sum(prediction | target).float()
