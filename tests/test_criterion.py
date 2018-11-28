import numpy as np
import torch

from unet3d.utils import DiceCoefficient, GeneralizedDiceLoss


class TestCriterion:
    def test_dice_coefficient(self):
        results = self._compute_criterion(DiceCoefficient())
        # check that all of the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)

    def test_generalized_dice_loss(self):
        results = self._compute_criterion(GeneralizedDiceLoss())
        # check that all of the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)

    @staticmethod
    def _compute_criterion(criterion):
        shape = [1, 0, 30, 30, 30]
        # channel size varies between 1 and 4
        results = []
        for C in range(1, 5):
            batch_shape = list(shape)
            batch_shape[1] = C
            batch_shape = tuple(batch_shape)
            # compute Dice Coefficient 100 times
            for i in range(100):
                dice = DiceCoefficient()
                input = torch.rand(batch_shape)
                target = torch.zeros(batch_shape).random_(0, 2)
                results.append(dice(input, target))

        return results
