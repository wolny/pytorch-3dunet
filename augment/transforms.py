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
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state):
        self.random_state = random_state

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, (1, 2))
        else:
            channels = [np.rot90(m[c], k, (1, 2)) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=15, axes=None):
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
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=0, mode='constant', cval=-1) for c in
                        range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class LabelToBoundary:
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the offsets (thickness) of the border as well as the axes (direction) across which the boundary
    will be computed via the convolution operator. The convolved images are stacked across the channel dimension (CxDxHxW)
    """

    AXES = {
        0: (0, 1, 2),
        1: (0, 2, 1),
        2: (2, 0, 1)
    }

    def __init__(self, axes, offsets, ignore_index=None):
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

        self.ignore_index = ignore_index

        self.kernels = []
        # create kernel for every axis-offset pair
        for axis in axes:
            for offset in offsets:
                self.kernels.append(self._create_kernel(axis, offset))

    @staticmethod
    def _create_kernel(axis, offset):
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
        :return: 4D binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """

        def _recover_ignore_index(input, mask):
            if mask is not None:
                input[mask] = self.ignore_index
            return input

        assert m.ndim == 3
        mask = None
        if self.ignore_index is not None:
            mask = m == self.ignore_index

        channels = [_recover_ignore_index(np.abs(convolve(m, k)), mask) for k in self.kernels]
        result = np.stack(channels, axis=0)
        result[result > 0] = 1
        return result


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
    seed = 47

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        self.mean = mean
        self.std = std
        self.phase = phase
        self.label_dtype = label_dtype

    def raw_transform(self):
        return Compose([
            Normalize(self.mean, self.std),
            ToTensor(expand_dims=True)
        ])

    def label_transform(self):
        return Compose([
            ToTensor(expand_dims=False, dtype=self.label_dtype)
        ])

    @classmethod
    def create(cls, mean, std, phase, label_dtype, **kwargs):
        return cls(mean, std, phase, label_dtype, **kwargs)


class StandardTransformer(BaseTransformer):
    """
    Standard data augmentation: random flips across randomly picked axis + random 90 degrees rotations.

    """

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()


class StandardTransformerWithWeights(StandardTransformer):
    def get_weight_transform(self):
        return super().label_transform()


class IsotropicRotationTransformer(BaseTransformer):
    """
    Data augmentation to be used with isotropic 3D volumes: random flips across randomly picked axis + random 90 deg
    rotations + random angle rotations across randomly picked axis.
    """

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()


class AnisotropicRotationTransformer(BaseTransformer):
    """
    Data augmentation to be used with anisotropic 3D volumes: random flips across randomly picked axis + random 90 deg
    rotations + random angle rotations across (1,0) axis.
    """

    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()


# FIXME: do not use random rotations (except RandomRotate90) together with LabelToBoundary, cause it will create
# artificial boundary signal; consider using different mode, e.g. mode='reflect' when doing rotations.
class LabelToBoundaryTransformer(BaseTransformer):
    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # this will give us 6 output channels with boundary signal
                LabelToBoundary(axes=(0, 1, 2), offsets=(1, 4), ignore_index=-1),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return Compose([
                LabelToBoundary(axes=(0, 1, 2), offsets=(1, 4), ignore_index=-1),
                # this will give us 6 output channels with boundary signal
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
