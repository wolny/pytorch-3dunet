import importlib

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure

from unet3d.losses import compute_per_channel_dice, expand_as_one_hot
from unet3d.utils import get_logger, adapted_rand

LOGGER = get_logger('EvalMetric')

SUPPORTED_METRICS = ['dice', 'iou', 'boundary_ap', 'dt_ap', 'quantized_dt_ap', 'angle', 'inverse_angular']


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-5, ignore_index=None, **kwargs):
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: Soft Dice Coefficient averaged over all channels/classes
        """
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]
        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        # batch dim must be 1
        input = input[0]
        target = target[0]
        assert input.size() == target.size()

        binary_prediction = self._binarize_predictions(input, n_classes)

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

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.sum(prediction | target).float()


class AdaptedRandError:
    def __init__(self, all_stats=False, **kwargs):
        self.all_stats = all_stats

    def __call__(self, input, target):
        return adapted_rand(input, target, all_stats=self.all_stats)


class BoundaryAdaptedRandError:
    def __init__(self, threshold=0.4, use_last_target=False, use_first_input=False, invert_pmaps=True, **kwargs):
        self.threshold = threshold
        self.use_last_target = use_last_target
        self.use_first_input = use_first_input
        self.invert_pmaps = invert_pmaps

    def __call__(self, input, target):
        if isinstance(input, torch.Tensor):
            assert input.dim() == 5
            # convert to numpy array
            input = input[0].detach().cpu().numpy()  # 4D

        if isinstance(target, torch.Tensor):
            if not self.use_last_target:
                assert target.dim() == 4
                # convert to numpy array
                target = target[0].detach().cpu().numpy()  # 3D
            else:
                # if use_last_target == True the target must be 5D (NxCxDxHxW)
                assert target.dim() == 5
                target = target[0, -1].detach().cpu().numpy()  # 3D

        if isinstance(input, np.ndarray):
            assert input.ndim == 4

        if isinstance(target, np.ndarray):
            assert target.ndim == 3

        if self.use_first_input:
            # compute only on the first input channel
            n_channels = 1
        else:
            n_channels = input.shape[0]

        per_channel_arand = []
        for c in range(n_channels):
            predictions = input[c]
            # threshold probability maps
            predictions = predictions > self.threshold

            if self.invert_pmaps:
                # for connected component analysis we need to treat boundary signal as background
                # assign 0-label to boundary mask
                predictions = np.logical_not(predictions)

            predictions = predictions.astype(np.uint8)
            # run connected components on the predicted mask; consider only 1-connectivity
            predicted = measure.label(predictions, background=0, connectivity=1)
            # make sure that target is 'int' type as well
            target = target.astype(np.int64)
            # compute AdaptedRand error
            arand = adapted_rand(predicted, target)
            per_channel_arand.append(arand)

        # get minimum AdaptedRand error across channels
        min_arand, c_index = np.min(per_channel_arand), np.argmin(per_channel_arand)
        LOGGER.info(f'Min AdaptedRand error: {min_arand}, channel: {c_index}')
        return min_arand


class _AbstractAP:
    def __init__(self, iou_range=(0.5, 1.0), ignore_index=-1, min_instance_size=None):
        self.iou_range = iou_range
        self.ignore_index = ignore_index
        self.min_instance_size = min_instance_size

    def __call__(self, input, target):
        raise NotImplementedError()

    def _calculate_average_precision(self, predicted, target, target_instances):
        recall, precision = self._roc_curve(predicted, target, target_instances)
        recall.insert(0, 0.0)  # insert 0.0 at beginning of list
        recall.append(1.0)  # insert 1.0 at end of list
        precision.insert(0, 0.0)  # insert 0.0 at beginning of list
        precision.append(0.0)  # insert 0.0 at end of list
        # make the precision(recall) piece-wise constant and monotonically decreasing
        # by iterating backwards starting from the last precision value (0.0)
        # see: https://www.jeremyjordan.me/evaluating-image-segmentation-models/ e.g.
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        # compute the area under precision recall curve by simple integration of piece-wise constant function
        ap = 0.0
        for i in range(1, len(recall)):
            ap += ((recall[i] - recall[i - 1]) * precision[i])
        return ap

    def _roc_curve(self, predicted, target, target_instances):
        ROC = []
        predicted, predicted_instances = self._filter_instances(predicted)

        # compute precision/recall curve points for various IoU values from a given range
        for min_iou in np.arange(self.iou_range[0], self.iou_range[1], 0.1):
            # initialize false negatives set
            false_negatives = set(target_instances)
            # initialize false positives set
            false_positives = set(predicted_instances)
            # initialize true positives set
            true_positives = set()

            for pred_label in predicted_instances:
                target_label = self._find_overlapping_target(pred_label, predicted, target, min_iou)
                if target_label is not None:
                    # update TP, FP and FN
                    if target_label == self.ignore_index:
                        # ignore if 'ignore_index' is the biggest overlapping
                        false_positives.discard(pred_label)
                    else:
                        true_positives.add(pred_label)
                        false_positives.discard(pred_label)
                        false_negatives.discard(target_label)

            tp = len(true_positives)
            fp = len(false_positives)
            fn = len(false_negatives)

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            ROC.append((recall, precision))

        # sort points by recall
        ROC = np.array(sorted(ROC, key=lambda t: t[0]))
        # return recall and precision values
        return list(ROC[:, 0]), list(ROC[:, 1])

    def _find_overlapping_target(self, predicted_label, predicted, target, min_iou):
        """
        Return ground truth label which overlaps by at least 'min_iou' with a given input label 'p_label'
        or None if such ground truth label does not exist.
        """
        mask_predicted = predicted == predicted_label
        overlapping_labels = target[mask_predicted]
        labels, counts = np.unique(overlapping_labels, return_counts=True)
        # retrieve the biggest overlapping label
        target_label_ind = np.argmax(counts)
        target_label = labels[target_label_ind]
        # return target label if IoU greater than 'min_iou'; since we're starting from 0.5 IoU there might be
        # only one target label that fulfill this criterion
        mask_target = target == target_label
        # return target_label if IoU > min_iou
        if self._iou(mask_predicted, mask_target) > min_iou:
            return target_label
        return None

    @staticmethod
    def _iou(prediction, target):
        """
        Computes intersection over union
        """
        intersection = np.logical_and(prediction, target)
        union = np.logical_or(prediction, target)
        return np.sum(intersection) / np.sum(union)

    def _filter_instances(self, input):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 'ignore_index'
        :param input: input instance segmentation
        :return: tuple: (instance segmentation with small instances filtered, set of unique labels without the 'ignore_index')
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    mask = input == label
                    input[mask] = self.ignore_index

        labels = set(np.unique(input))
        labels.discard(self.ignore_index)
        return input, labels

    @staticmethod
    def _dt_to_cc(distance_transform, threshold):
        """
        Threshold a given distance_transform and returns connected components.
        :param distance_transform: 3D distance transform matrix
        :param threshold: threshold energy level
        :return: 3D segmentation volume
        """
        boundary = (distance_transform > threshold).astype(np.uint8)
        return measure.label(boundary, background=0, connectivity=1)


class StandardAveragePrecision(_AbstractAP):
    def __init__(self, iou_range=(0.5, 1.0), ignore_index=-1, min_instance_size=None, **kwargs):
        super().__init__(iou_range, ignore_index, min_instance_size)

    def __call__(self, input, target):
        assert isinstance(input, np.ndarray) and isinstance(target, np.ndarray)
        assert input.ndim == target.ndim == 3

        target, target_instances = self._filter_instances(target)

        return self._calculate_average_precision(input, target, target_instances)


class DistanceTransformAveragePrecision(_AbstractAP):
    def __init__(self, threshold=0.1, **kwargs):
        super().__init__()
        self.threshold = threshold

    def __call__(self, input, target):
        if isinstance(input, torch.Tensor):
            assert input.dim() == 5
            # convert to numpy array
            input = input[0, 0].detach().cpu().numpy()  # 3D distance transform

        if isinstance(target, torch.Tensor):
            assert target.dim() == 5
            target = target[0, 0].detach().cpu().numpy()  # 3D distance transform

        if isinstance(input, np.ndarray):
            assert input.ndim == 3

        if isinstance(target, np.ndarray):
            assert target.ndim == 3

        predicted_cc = self._dt_to_cc(input, self.threshold)
        target_cc = self._dt_to_cc(target, self.threshold)

        # get ground truth label set
        target_cc, target_instances = self._filter_instances(target_cc)

        return self._calculate_average_precision(predicted_cc, target_cc, target_instances)


class QuantizedDistanceTransformAveragePrecision(_AbstractAP):
    def __init__(self, threshold=0, **kwargs):
        super().__init__()
        self.threshold = threshold

    def __call__(self, input, target):
        if isinstance(input, torch.Tensor):
            assert input.dim() == 5
            # convert probability maps to label tensor
            input = torch.argmax(input[0], dim=0)
            # convert to numpy array
            input = input.detach().cpu().numpy()  # 3D distance transform

        if isinstance(target, torch.Tensor):
            assert target.dim() == 4
            target = target[0].detach().cpu().numpy()  # 3D distance transform

        if isinstance(input, np.ndarray):
            assert input.ndim == 3

        if isinstance(target, np.ndarray):
            assert target.ndim == 3

        predicted_cc = self._dt_to_cc(input, self.threshold)
        target_cc = self._dt_to_cc(target, self.threshold)

        # get ground truth label set
        target_cc, target_instances = self._filter_instances(target_cc)

        return self._calculate_average_precision(predicted_cc, target_cc, target_instances)


class BoundaryAveragePrecision(_AbstractAP):
    """
    Computes Average Precision given boundary prediction and ground truth instance segmentation.
    """

    def __init__(self, threshold=0.4, iou_range=(0.5, 1.0), ignore_index=-1, min_instance_size=None,
                 use_last_target=False, **kwargs):
        """
        :param threshold: probability value at which the input is going to be thresholded
        :param iou_range: compute ROC curve for the the range of IoU values: range(min,max,0.05)
        :param ignore_index: label to be ignored during computation
        :param min_instance_size: minimum size of the predicted instances to be considered
        :param use_last_target: if True use the last target channel to compute AP
        """
        super().__init__(ignore_index, min_instance_size, iou_range)
        self.threshold = threshold
        # always have well defined ignore_index
        if ignore_index is None:
            ignore_index = -1
        self.iou_range = iou_range
        self.ignore_index = ignore_index
        self.min_instance_size = min_instance_size
        self.use_last_target = use_last_target

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW) / or 4D numpy.ndarray
        :param target: 4D or 5D ground truth instance segmentation torch long tensor / or 3D numpy.ndarray
        :return: highest average precision among channels
        """
        if isinstance(input, torch.Tensor):
            assert input.dim() == 5
            # convert to numpy array
            input = input[0].detach().cpu().numpy()  # 4D

        if isinstance(target, torch.Tensor):
            if not self.use_last_target:
                assert target.dim() == 4
                # convert to numpy array
                target = target[0].detach().cpu().numpy()  # 3D
            else:
                # if use_last_target == True the target must be 5D (NxCxDxHxW)
                assert target.dim() == 5
                target = target[0, -1].detach().cpu().numpy()  # 3D

        if isinstance(input, np.ndarray):
            assert input.ndim == 4

        if isinstance(target, np.ndarray):
            assert target.ndim == 3

        # filter small instances from the target and get ground truth label set (without 'ignore_index')
        target, target_instances = self._filter_instances(target)

        per_channel_ap = []
        n_channels = input.shape[0]
        for c in range(n_channels):
            predictions = input[c]
            # threshold probability maps
            predictions = predictions > self.threshold
            # for connected component analysis we need to treat boundary signal as background
            # assign 0-label to boundary mask
            predictions = np.logical_not(predictions).astype(np.uint8)
            # run connected components on the predicted mask; consider only 1-connectivity
            predicted = measure.label(predictions, background=0, connectivity=1)
            ap = self._calculate_average_precision(predicted, target, target_instances)
            per_channel_ap.append(ap)

        # get maximum average precision across channels
        max_ap, c_index = np.max(per_channel_ap), np.argmax(per_channel_ap)
        LOGGER.info(f'Max average precision: {max_ap}, channel: {c_index}')
        return max_ap


class WithinAngleThreshold:
    """
    Returns the percentage of predicted directions which are more than 'angle_threshold' apart from the ground
    truth directions. 'angle_threshold' is expected to be given in degrees not radians.
    """

    def __init__(self, angle_threshold, **kwargs):
        self.threshold_radians = angle_threshold / 360 * np.pi

    def __call__(self, inputs, targets):
        assert isinstance(inputs, list)
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets)

        within_count = 0
        total_count = 0
        for input, target in zip(inputs, targets):
            # normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
            stability_coeff = 0.999999
            input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
            target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
            # compute cosine map
            cosines = (input * target).sum(dim=1)
            error_radians = torch.acos(cosines)
            # increase by the number of directions within the threshold
            within_count += error_radians[error_radians < self.threshold_radians].numel()
            # increase by the number of all directions
            total_count += error_radians.numel()

        return torch.tensor(within_count / total_count)


class InverseAngularError:
    def __init__(self, **kwargs):
        pass

    def __call__(self, inputs, targets, **kwargs):
        assert isinstance(inputs, list)
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets)

        total_error = 0
        for input, target in zip(inputs, targets):
            # normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
            stability_coeff = 0.999999
            input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
            target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
            # compute cosine map
            cosines = (input * target).sum(dim=1)
            error_radians = torch.acos(cosines)
            total_error += error_radians.sum()

        return torch.tensor(1. / total_error)


class PSNR:
    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        assert input.size() == target.size()

        return 10 * torch.log10(1 / torch.max(F.mse_loss(input, target), torch.tensor(0.01).to(input.device)))


def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('unet3d.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)
