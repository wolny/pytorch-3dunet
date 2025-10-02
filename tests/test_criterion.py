import numpy as np
import torch
import torch.nn as nn
from skimage import measure

from pytorch3dunet.augment.transforms import LabelToAffinities, StandardLabelToBoundary
from pytorch3dunet.unet3d.losses import (
    BCEDiceLoss,
    DiceLoss,
    GeneralizedDiceLoss,
    MaskingLossWrapper,
    SkipLastTargetChannelWrapper,
    WeightedSmoothL1Loss,
)
from pytorch3dunet.unet3d.metrics import (
    AdaptedRandError,
    BoundaryAdaptedRandError,
    BoundaryAveragePrecision,
    DiceCoefficient,
    MeanIoU,
)


def _compute_criterion(criterion, n_times=100):
    shape = [1, 0, 30, 30, 30]
    # channel size varies between 1 and 4
    results = []
    for C in range(1, 5):
        batch_shape = list(shape)
        batch_shape[1] = C
        batch_shape = tuple(batch_shape)
        results.append(_eval_criterion(criterion, batch_shape, n_times))

    return results


def _eval_criterion(criterion, batch_shape, n_times=100):
    with torch.no_grad():
        results = []
        # compute criterion n_times
        for _ in range(n_times):
            input = torch.rand(batch_shape)
            target = torch.zeros(batch_shape).random_(0, 2)
            output = criterion(input, target)
            results.append(output)

    return results


class TestCriterion:
    def test_dice_coefficient(self):
        results = _compute_criterion(DiceCoefficient())
        # check that all the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)

    def test_mean_iou_simple(self):
        results = _compute_criterion(MeanIoU())
        # check that all of the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)

    def test_mean_iou_multi_channel(self):
        criterion = MeanIoU()
        pred = torch.rand(10, 3, 10, 10, 10)
        target = pred > 0.5
        target = target.long()
        assert criterion(pred, target) == 1

    def test_mean_iou_multi_class(self):
        criterion = MeanIoU()
        n_classes = 5
        n_batch = 10
        pred = torch.rand(n_batch, n_classes, 10, 10, 10)
        target = torch.randint(0, n_classes, (n_batch, 10, 10, 10))
        mean_iou = criterion(pred, target)
        assert mean_iou >= 0

    def test_average_precision_synthethic_data(self):
        input = np.zeros((64, 200, 200), dtype=np.int32)
        for i in range(40, 200, 40):
            input[:, :, i : i + 2] = 1
        for i in range(40, 200, 40):
            input[:, i : i + 2, :] = 1
        for i in range(40, 64, 40):
            input[i : i + 2, :, :] = 1

        target = measure.label(np.logical_not(input).astype(np.int32), background=0)
        input = torch.tensor(input.reshape((1, 1) + input.shape))
        target = torch.tensor(target.reshape((1, 1) + target.shape))
        ap = BoundaryAveragePrecision()
        # expect perfect AP
        assert ap(input, target) == 1.0

    def test_average_precision_real_data(self, ovule_label):
        label = ovule_label[64:128, 64:128, 64:128]
        ltb = LabelToAffinities((1, 2, 4, 6), aggregate_affinities=True)
        pred = ltb(label)
        label = torch.tensor(label.reshape((1, 1) + label.shape).astype("int64"))
        pred = torch.tensor(np.expand_dims(pred, 0))
        ap = BoundaryAveragePrecision()
        assert ap(pred, label) > 0.5

    def test_adapted_rand_error(self, ovule_label):
        label = ovule_label[64:128, 64:128, 64:128].astype("int64")
        input = torch.tensor(label.reshape((1, 1) + label.shape))
        label = torch.tensor(label.reshape((1, 1) + label.shape))
        arand = AdaptedRandError()
        assert arand(input, label) == 0

    def test_adapted_rand_error_on_real_data(self, ovule_label):
        label = ovule_label[64:128, 64:128, 64:128].astype("int64")
        ltb = StandardLabelToBoundary()
        pred = ltb(label)
        label = torch.tensor(label.reshape((1, 1) + label.shape))
        pred = torch.tensor(np.expand_dims(pred, 0))
        arand = BoundaryAdaptedRandError(use_last_target=True)
        assert arand(pred, label) < 0.2

    def test_generalized_dice_loss(self):
        results = _compute_criterion(GeneralizedDiceLoss())
        # check that all of the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)

    def test_dice_loss(self):
        results = _compute_criterion(DiceLoss())
        # check that all of the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)

    def test_bce_dice_loss(self):
        results = _compute_criterion(BCEDiceLoss(1.0))
        results = np.array(results)
        assert np.all(results > 0)

    def test_ignore_index_loss(self):
        loss = MaskingLossWrapper(nn.BCEWithLogitsLoss(), ignore_index=-1)
        input = torch.rand((3, 3))
        input[1, 1] = 1.0
        input.requires_grad = True
        target = -1.0 * torch.ones((3, 3))
        target[1, 1] = 1.0
        output = loss(input, target)
        output.backward()

    def test_skip_last_target_channel(self):
        loss = SkipLastTargetChannelWrapper(nn.BCEWithLogitsLoss())
        input = torch.rand(1, 1, 3, 3, 3, requires_grad=True)
        target = torch.empty(1, 2, 3, 3, 3).random_(2)
        output = loss(input, target)
        output.backward()
        assert output.item() > 0

    def test_weighted_smooth_l1loss(self):
        loss_criterion = WeightedSmoothL1Loss(threshold=0.0, initial_weight=0.1)
        input = torch.randn(3, 16, 64, 64, 64, requires_grad=True)
        target = torch.randn(3, 16, 64, 64, 64)
        loss = loss_criterion(input, target)
        loss.backward()
        assert loss > 0
