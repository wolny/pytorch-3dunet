import importlib

import numpy as np
import torch
from skimage import measure
from skimage.metrics import adapted_rand_error, peak_signal_noise_ratio, mean_squared_error

from pytorch3dunet.unet3d.losses import compute_per_channel_dice
from pytorch3dunet.unet3d.seg_metrics import AveragePrecision, Accuracy
from pytorch3dunet.unet3d.utils import get_logger, convert_to_numpy

logger = get_logger('EvalMetric')


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and then simply taking the average.
    Input is expected to be probabilities instead of logits.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.

    Args:
        skip_background (bool): if True, background class (i.e. 0-label) will be skipped when computing IoU
    """

    def __init__(self, skip_background=True, **kwargs):
        self.skip_background = skip_background

    def __call__(self, input, target):
        """Compute mean IoU.

        Args:
            input: 5D probability maps torch float tensor (NxCxDxHxW).
            target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot.

        Returns:
            Intersection over union averaged over all channels.
        """
        assert input.dim() == 5

        n_classes = input.size(1)

        if target.dim() == 4:
            # convert input to segmentation
            input = input.argmax(dim=1)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            # convert target to byte
            _target = _target.byte()
            per_channel_iou = []
            start_idx = 0
            # skip background only if target is 4D; for channel-wise computation (i.e. if target is 5D) we need to include it
            if self.skip_background and target.dim() == 4:
                start_idx = 1

            for c in range(start_idx, n_classes):
                if target.dim() == 5:
                    iou = self._jaccard_index(_input[c] > 0.5, _target[c])
                    per_channel_iou.append(iou)
                else:
                    iou = self._jaccard_index(_input == c, _target == c)
                    per_channel_iou.append(iou)

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.tensor(per_channel_iou).mean()
            per_batch_iou.append(mean_iou)

        return torch.tensor(per_batch_iou).mean()

    def _jaccard_index(self, prediction, target):
        """Computes IoU for a given target and prediction tensors.

        Args:
            prediction: Predicted binary mask.
            target: Ground truth binary mask.

        Returns:
            Intersection over union score.
        """
        epsilon = 1e-8
        intersection = torch.logical_and(target, prediction).sum()
        union = torch.logical_or(target, prediction).sum()
        return (intersection + epsilon) / (union + epsilon)


class AdaptedRandError:
    """
    A functor which computes an Adapted Rand error as defined by the SNEMI3D contest
    (http://brainiac2.mit.edu/SNEMI3D/evaluation).

    This is a generic implementation which takes the input, converts it to the segmentation image (see `input_to_segm()`)
    and then computes the ARand between the segmentation and the ground truth target. Depending on one's use case
    it's enough to extend this class and implement the `input_to_segm` method.

    Args:
        use_last_target (bool): if true, use the last channel from the target to compute the ARand, otherwise the first.
    """

    def __init__(self, use_last_target=False, ignore_index=None, **kwargs):
        self.use_last_target = use_last_target
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        """
        Compute ARand Error for each input, target pair in the batch and return the mean value.

        Args:
            input (torch.tensor):  5D (NCDHW) output from the network
            target (torch.tensor): 5D (NCDHW) ground truth segmentation

        Returns:
            average ARand Error across the batch
        """

        # converts input and target to numpy arrays
        input, target = convert_to_numpy(input, target)
        if self.use_last_target:
            target = target[:, -1, ...]  # 4D
        else:
            # use 1st target channel
            target = target[:, 0, ...]  # 4D

        # ensure target is of integer type
        target = target.astype(np.int32)

        if self.ignore_index is not None:
            target[target == self.ignore_index] = 0

        per_batch_arand = []
        for _input, _target in zip(input, target):
            if np.all(_target == _target.flat[0]):  # skip ARand eval if there is only one label in the patch due to zero-division
                logger.info('Skipping ARandError computation: only 1 label present in the ground truth')
                per_batch_arand.append(0.)
                continue

            # convert _input to segmentation CDHW
            segm = self.input_to_segm(_input)
            assert segm.ndim == 4

            # compute per channel arand and return the minimum value
            per_channel_arand = [adapted_rand_error(_target, channel_segm)[0] for channel_segm in segm]
            per_batch_arand.append(np.min(per_channel_arand))

        # return mean arand error
        mean_arand = torch.mean(torch.tensor(per_batch_arand))
        logger.info(f'ARand: {mean_arand.item()}')
        return mean_arand

    def input_to_segm(self, input):
        """Converts input tensor to segmentation image.

        If the input is boundary pmaps, one option would be to threshold it and run connected
        components in order to return the segmentation.

        Args:
            input: 4D tensor (CDHW).

        Returns:
            Segmentation volume, 4D (segmentation per channel).
        """
        # by deafult assume that input is a segmentation volume itself
        return input


class BoundaryAdaptedRandError(AdaptedRandError):
    """
    Compute ARand between the input boundary map and target segmentation.
    Boundary map is thresholded, and connected components is run to get the predicted segmentation
    """

    def __init__(self, thresholds=None, use_last_target=True, ignore_index=None, input_channel=None, invert_pmaps=True,
                 save_plots=False, plots_dir='.', **kwargs):
        super().__init__(use_last_target=use_last_target, ignore_index=ignore_index, save_plots=save_plots,
                         plots_dir=plots_dir, **kwargs)

        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel
        self.invert_pmaps = invert_pmaps

    def input_to_segm(self, input):
        if self.input_channel is not None:
            input = np.expand_dims(input[self.input_channel], axis=0)

        segs = []
        for predictions in input:
            for th in self.thresholds:
                # threshold probability maps
                predictions_th = predictions > th

                if self.invert_pmaps:
                    # for connected component analysis we need to treat boundary signal as background
                    # assign 0-label to boundary mask
                    predictions_th = np.logical_not(predictions_th)

                predictions_th = predictions_th.astype(np.uint8)
                # run connected components on the predicted mask; consider only 1-connectivity
                seg = measure.label(predictions_th, background=0, connectivity=1)
                segs.append(seg)

        return np.stack(segs)


class GenericAdaptedRandError(AdaptedRandError):
    def __init__(self, input_channels, thresholds=None, use_last_target=True, ignore_index=None, invert_channels=None,
                 **kwargs):

        super().__init__(use_last_target=use_last_target, ignore_index=ignore_index, **kwargs)
        assert isinstance(input_channels, list) or isinstance(input_channels, tuple)
        self.input_channels = input_channels
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        if invert_channels is None:
            invert_channels = []
        self.invert_channels = invert_channels

    def input_to_segm(self, input):
        # pick only the channels specified in the input_channels
        results = []
        for i in self.input_channels:
            c = input[i]
            # invert channel if necessary
            if i in self.invert_channels:
                c = 1 - c
            results.append(c)

        input = np.stack(results)

        segs = []
        for predictions in input:
            for th in self.thresholds:
                # run connected components on the predicted mask; consider only 1-connectivity
                seg = measure.label((predictions > th).astype(np.uint8), background=0, connectivity=1)
                segs.append(seg)

        return np.stack(segs)


class GenericAveragePrecision:
    def __init__(self, min_instance_size=None, use_last_target=False, metric='ap', **kwargs):
        self.min_instance_size = min_instance_size
        self.use_last_target = use_last_target
        assert metric in ['ap', 'acc']
        if metric == 'ap':
            # use AveragePrecision
            self.metric = AveragePrecision()
        else:
            # use Accuracy at 0.5 IoU
            self.metric = Accuracy(iou_threshold=0.5)

    def __call__(self, input, target):
        if target.dim() == 5:
            if self.use_last_target:
                target = target[:, -1, ...]  # 4D
            else:
                # use 1st target channel
                target = target[:, 0, ...]  # 4D

        input1 = input2 = input
        multi_head = isinstance(input, tuple)
        if multi_head:
            input1, input2 = input

        input1, input2, target = convert_to_numpy(input1, input2, target)

        batch_aps = []
        i_batch = 0
        # iterate over the batch
        for inp1, inp2, tar in zip(input1, input2, target):
            if multi_head:
                inp = (inp1, inp2)
            else:
                inp = inp1

            segs = self.input_to_seg(inp, tar)  # expects 4D
            assert segs.ndim == 4
            # convert target to seg
            tar = self.target_to_seg(tar)

            # filter small instances if necessary
            tar = self._filter_instances(tar)

            # compute average precision per channel
            segs_aps = [self.metric(self._filter_instances(seg), tar) for seg in segs]

            logger.info(f'Batch: {i_batch}. Max Average Precision for channel: {np.argmax(segs_aps)}')
            # save max AP
            batch_aps.append(np.max(segs_aps))
            i_batch += 1

        return torch.tensor(batch_aps).mean()

    def _filter_instances(self, input):
        """Filters instances smaller than 'min_instance_size' by overriding them with 0-index.

        Args:
            input: Input instance segmentation.

        Returns:
            Filtered instance segmentation.
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    input[input == label] = 0
        return input

    def input_to_seg(self, input, target=None):
        raise NotImplementedError

    def target_to_seg(self, target):
        return target


class BlobsAveragePrecision(GenericAveragePrecision):
    """Computes Average Precision given foreground prediction and ground truth instance segmentation.

    Args:
        thresholds: List of thresholds for segmentation. Default: [0.4, 0.5, 0.6, 0.7, 0.8].
        metric: Metric to use ('ap' or 'acc'). Default: 'ap'.
        min_instance_size: Minimum instance size to consider.
        input_channel: Channel to use from input. Default: 0.
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class BlobsBoundaryAveragePrecision(GenericAveragePrecision):
    """Computes Average Precision given foreground and boundary predictions.

    Segmentation mask is computed as (P_mask - P_boundary) > th followed by a connected component.

    Args:
        thresholds: List of thresholds for segmentation. Default: [0.3, 0.4, 0.5, 0.6, 0.7].
        metric: Metric to use ('ap' or 'acc'). Default: 'ap'.
        min_instance_size: Minimum instance size to consider.
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds

    def input_to_seg(self, input, target=None):
        # input = P_mask - P_boundary
        input = input[0] - input[1]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class BoundaryAveragePrecision(GenericAveragePrecision):
    """Computes Average Precision given boundary prediction and ground truth instance segmentation.

    Args:
        thresholds: List of thresholds for segmentation. Default: [0.3, 0.4, 0.5, 0.6].
        min_instance_size: Minimum instance size to consider.
        input_channel: Channel to use from input. Default: 0.
    """

    def __init__(self, thresholds=None, min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True)
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            seg = measure.label(np.logical_not(input > th).astype(np.uint8), background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class PSNR:
    """Computes Peak Signal to Noise Ratio.

    Use e.g. as an eval metric for denoising task.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return peak_signal_noise_ratio(target, input)


class MSE:
    """Computes MSE between input and target."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return mean_squared_error(input, target)


def get_evaluation_metric(config):
    """Returns the evaluation metric function based on provided configuration.

    Args:
        config: A top level configuration object containing the 'eval_metric' key.

    Returns:
        An instance of the evaluation metric.
    """

    def _metric_class(class_name):
        m = importlib.import_module('pytorch3dunet.unet3d.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)
