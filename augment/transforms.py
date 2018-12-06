import numpy as np
import torch
from scipy.ndimage import rotate
from scipy.ndimage.filters import convolve
from torchvision.transforms import Compose


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


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=45, axes=None):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=0, mode='constant', cval=-1)
        else:
            for c in range(m.shape[0]):
                m[c] = rotate(m[c], angle, axes=axis, reshape=False, order=0, mode='constant', cval=-1)

        return m


class LabelToBoundary:
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the offsets (thickness) of the border as well as the axes (direction) across which the boundary
    will be computed via the convolution operator. The convolved images are averaged and the final boundary
    is placed where the average value is larger than 0.5
    """

    AXES = {
        0: (0, 1, 2),
        1: (0, 2, 1),
        2: (2, 0, 1)
    }

    def __init__(self, axes, offsets):
        if isinstance(axes, int):
            assert axes in [0, 1, 2], "Axis must be one of [0, 1, 2]"
            axes = [self.AXES[axes]]
        elif isinstance(axes, list) or isinstance(axes, tuple):
            assert all(a in [0, 1, 2] for a in axes), "Axis must be one of [0, 1, 2]"
            assert len(set(axes)) == len(axes), "'axes' must be unique"
            axes = [self.AXES[a] for a in axes]
        else:
            raise ValueError(f"Unsupported 'axes' type {type(axes)}")

        if isinstance(offsets, int):
            assert offsets > 0, "'offset' must be positive"
            offsets = [offsets]
        elif isinstance(offsets, list) or isinstance(offsets, tuple):
            assert all(a > 0 for a in offsets), "'offset' must be positive"
            assert len(set(offsets)) == len(offsets), "'offsets' must be unique"
        else:
            raise ValueError(f"Unsupported 'offsets' type {type(offsets)}")

        self.kernels = []
        # create kernel for every axis-offset pair
        for axis in axes:
            for offset in offsets:
                self.kernels.append(self._create_kernel(axis, offset))

    def _create_kernel(self, axis, offset):
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, 1, k_size), dtype=np.int)
        k[0, 0, 0] = 1
        k[0, 0, offset] = -1
        return np.transpose(k, axis)

    def __call__(self, m):
        """
        Extract boundaries from a given 3D label tensor.
        :param m: input 3D tensor
        :return: 3D binary mask of the same size as 'm', with 1-label corresponding to the boundary
            and 0-label corresponding to the background
        """
        assert m.ndim == 3
        result = np.zeros_like(m, dtype=np.float)
        for kernel in self.kernels:
            # convolve the input tensor with a given kernel
            convolved = np.abs(convolve(m, kernel))
            # convert to binary mask
            convolved[convolved > 0] = 1
            # accumulate
            result = result + convolved
        # compute the average
        result = result / len(self.kernels)
        # threshold
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        return result.astype(m.dtype)


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


# Helper Transformer classes

class BaseTransformer:
    """
    Base transformer class used for data augmentation.
    """

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        self.mean = mean
        self.std = std
        self.phase = phase
        self.label_dtype = label_dtype

    def get_transforms(self):
        """
        Returns transforms for both raw and label patches. It's up to the implementor to make the transforms consistent
        between raw and labels.
        :return: (raw_transform, label_transform)
        """
        raw_transform = Compose([
            Normalize(self.mean, self.std),
            ToTensor(expand_dims=True)
        ])
        label_transform = Compose([
            ToTensor(expand_dims=False, dtype=self.label_dtype)
        ])

        return raw_transform, label_transform

    @classmethod
    def create(cls, mean, std, phase, label_dtype, **kwargs):
        return cls(mean, std, phase, label_dtype, **kwargs)


class StandardTransformer(BaseTransformer):
    def get_transforms(self):
        seed = 47
        if self.phase == 'train':
            raw_transform = Compose([
                Normalize(self.mean, self.std),
                RandomFlip(np.random.RandomState(seed)),
                RandomRotate90(np.random.RandomState(seed)),
                ToTensor(expand_dims=True)
            ])
            label_transform = Compose([
                RandomFlip(np.random.RandomState(seed)),
                RandomRotate90(np.random.RandomState(seed)),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
            return raw_transform, label_transform
        else:
            return super().get_transforms()


class ExtendedTransformer(BaseTransformer):
    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def get_transforms(self):
        seed = 47
        if self.phase == 'train':
            raw_transform = Compose([
                Normalize(self.mean, self.std),
                RandomFlip(np.random.RandomState(seed)),
                RandomRotate90(np.random.RandomState(seed)),
                RandomRotate(np.random.RandomState(seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=True)
            ])
            label_transform = Compose([
                RandomFlip(np.random.RandomState(seed)),
                RandomRotate90(np.random.RandomState(seed)),
                RandomRotate(np.random.RandomState(seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
            return raw_transform, label_transform
        else:
            return super().get_transforms()
