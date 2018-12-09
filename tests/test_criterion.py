import numpy as np
import pytest
import torch
import torch.nn as nn

from unet3d.losses import DiceCoefficient, GeneralizedDiceLoss, WeightedCrossEntropyLoss, IgnoreIndexLossWrapper


def _compute_criterion(criterion, n_times=100):
    shape = [1, 0, 30, 30, 30]
    # channel size varies between 1 and 4
    results = []
    for C in range(1, 5):
        batch_shape = list(shape)
        batch_shape[1] = C
        batch_shape = tuple(batch_shape)
        results = results + _eval_criterion(criterion, batch_shape, n_times)

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

    def test_generalized_dice_loss(self):
        results = _compute_criterion(GeneralizedDiceLoss())
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
            IgnoreIndexLossWrapper(nn.CrossEntropyLoss())

    def test_ignore_index_loss(self):
        loss = IgnoreIndexLossWrapper(nn.BCELoss(), ignore_index=-1)
        input = torch.zeros((3, 3))
        input[1, 1] = 1.
        target = -1. * torch.ones((3, 3))
        target[1, 1] = 1.
        output = loss(input, target)
        assert output.item() == 0

    def test_ignore_index_loss_with_dice_coeff(self):
        loss = IgnoreIndexLossWrapper(DiceCoefficient(), ignore_index=-1)
        input = torch.zeros((3, 3))
        input[1, 1] = 1.
        target = -1. * torch.ones((3, 3))
        target[1, 1] = 1.

        actual = loss(input, target)

        target = input.clone()
        expected = loss(input, target)

        assert expected == actual
