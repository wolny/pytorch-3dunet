from itertools import combinations

import numpy as np
import torch


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > 0.5:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    for c in range(m.shape[0]):
                        m[c] = np.flip(m[c], axis)

        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state):
        self.random_state = random_state

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 3)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, (1, 2))
        else:
            for c in range(m.shape[0]):
                m[c] = np.rot90(m[c], k, (1, 2))

        return m

    @staticmethod
    def _rot_planes(ndim=3):
        return list(combinations(range(ndim), 2))


class Normalize:
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean, std, eps=1e-4):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        return (m - self.mean) / (self.std + self.eps)


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Identity:
    def __call__(self, m):
        return m
