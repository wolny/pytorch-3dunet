import numpy as np
import torch

from unet3d.utils import DiceCoefficient


class TestCriterion:
    def test_dice_coefficient(self):
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

        # check that all of the coefficients belong to [0, 1]
        results = np.array(results)
        assert np.all(results > 0)
        assert np.all(results < 1)
