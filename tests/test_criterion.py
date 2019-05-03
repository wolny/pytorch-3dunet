import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
from skimage import measure

from augment.transforms import LabelToAffinities, StandardLabelToBoundary
from unet3d.losses import GeneralizedDiceLoss, WeightedCrossEntropyLoss, BCELossWrapper, \
    DiceLoss, TagsAngularLoss
from unet3d.metrics import DiceCoefficient, MeanIoU, BoundaryAveragePrecision, AdaptedRandError, \
    BoundaryAdaptedRandError


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
    results = []
    # compute criterion n_times
    for i in range(n_times):
        input = torch.rand(batch_shape)
        target = torch.zeros(batch_shape).random_(0, 2)
        results.append(criterion(input, target))

    return results


class TestCriterion:
    def test_dice_coefficient(self):
        results = _compute_criterion(DiceCoefficient())
        # check that all of the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)

    def test_mean_iou_simple(self):
        results = _compute_criterion(MeanIoU())
        # check that all of the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)

    def test_mean_iou(self):
        criterion = MeanIoU()
        x = torch.randn(3, 3, 3, 3)
        _, index = torch.max(x, dim=0, keepdim=True)
        # create target tensor
        target = torch.zeros_like(x, dtype=torch.long).scatter_(0, index, 1)
        pred = torch.zeros_like(target, dtype=torch.float)
        mask = target == 1
        # create prediction tensor
        pred[mask] = torch.rand(1)
        # make sure the dimensions are right
        target = torch.unsqueeze(target, dim=0)
        pred = torch.unsqueeze(pred, dim=0)
        assert criterion(pred, target) == 1

    def test_mean_iou_one_channel(self):
        criterion = MeanIoU()
        pred = torch.rand(1, 1, 3, 3, 3)
        target = pred > 0.5
        target = target.long()
        assert criterion(pred, target) == 1

    def test_average_precision_synthethic_data(self):
        input = np.zeros((64, 200, 200), dtype=np.int)
        for i in range(40, 200, 40):
            input[:, :, i:i + 2] = 1
        for i in range(40, 200, 40):
            input[:, i:i + 2, :] = 1
        for i in range(40, 64, 40):
            input[i:i + 2, :, :] = 1

        target = measure.label(np.logical_not(input).astype(np.int), background=0)
        input = np.expand_dims(input, axis=0)
        ap = BoundaryAveragePrecision()
        # expect perfect AP
        assert ap(input, target) == 1.0

    def test_average_precision_real_data(self):
        l_file = 'resources/sample_patch.h5'
        with h5py.File(l_file, 'r') as f:
            label = f['big_label'][...]
            ltb = LabelToAffinities((1, 2, 4, 6), aggregate_affinities=True)
            pred = ltb(label)
            # don't compare instances smaller than 25K voxels
            ap = BoundaryAveragePrecision(min_instance_size=25000)
            assert ap(pred, label) > 0.5

    def test_adapted_rand_error(self):
        l_file = 'resources/sample_patch.h5'
        with h5py.File(l_file, 'r') as f:
            label = f['big_label'][...]
            arand = AdaptedRandError()
            assert arand(label, label) == 0

    def test_adapted_rand_error_on_real_data(self):
        l_file = 'resources/sample_patch.h5'
        with h5py.File(l_file, 'r') as f:
            label = f['big_label'][...]
            ltb = StandardLabelToBoundary()
            pred = ltb(label)
            arand = BoundaryAdaptedRandError(all_stats=True)
            assert arand(pred, label) < 0.5

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

    def test_weighted_generalized_dice_loss(self):
        shape = [1, 0, 30, 30, 30]
        results = []
        for C in range(1, 5):
            batch_shape = list(shape)
            batch_shape[1] = C
            batch_shape = tuple(batch_shape)
            results = results + _eval_criterion(GeneralizedDiceLoss(weight=torch.rand(C)), batch_shape)

        # check that all of the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)

    def test_weighted_ce(self):
        criterion = WeightedCrossEntropyLoss()
        shape = [1, 0, 30, 30, 30]
        target_shape = [1, 30, 30, 30]
        results = []
        for C in range(1, 5):
            input_shape = list(shape)
            input_shape[1] = C
            input_shape = tuple(input_shape)
            for i in range(100):
                input = torch.rand(input_shape)
                target = torch.zeros(target_shape, dtype=torch.long).random_(0, C)
                results.append(criterion(input, target))

        results = np.array(results)
        assert np.all(results >= 0)

    def test_ignore_index_loss_wrapper_unsupported_loss(self):
        with pytest.raises(RuntimeError):
            BCELossWrapper(nn.CrossEntropyLoss())

    def test_ignore_index_loss(self):
        loss = BCELossWrapper(nn.BCELoss(), ignore_index=-1)
        input = torch.zeros((3, 3))
        input[1, 1] = 1.
        target = -1. * torch.ones((3, 3))
        target[1, 1] = 1.
        output = loss(input, target)
        assert output.item() == 0

    def test_ignore_index_loss_backward(self):
        loss = BCELossWrapper(nn.BCELoss(), ignore_index=-1)
        input = torch.zeros((3, 3), requires_grad=True)
        target = -1. * torch.ones((3, 3))
        output = loss(input, target)
        output.backward()
        assert output.item() == 0

    def test_ignore_index_loss_with_dice_coeff(self):
        loss = DiceCoefficient(ignore_index=-1)
        input = torch.zeros((3, 3))
        input[1, 1] = 1.
        target = -1. * torch.ones((3, 3))
        target[1, 1] = 1.

        actual = loss(input, target)

        target = input.clone()
        expected = loss(input, target)

        assert expected == actual

    def test_tags_angular_loss(self):
        coeff = [1.0, 0.8, 0.5]
        loss_criterion = TagsAngularLoss(coeff)
        inputs = [torch.rand((1, 3, 4, 4, 4)) for _ in range(len(coeff))]
        inputs = [i / torch.norm(i, p=2, dim=1).clamp(min=1e-8) for i in inputs]
        targets = [torch.rand((1, 3, 4, 4, 4)) for _ in range(len(coeff))]
        targets = [i / torch.norm(i, p=2, dim=1).clamp(min=1e-8) for i in targets]

        loss = loss_criterion(inputs, targets, None)
        assert loss > 0
