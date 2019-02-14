import random

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

    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='constant'):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=0, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=0, mode=self.mode, cval=-1) for c in
                        range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast(object):
    """
       Adjust the contrast of an image by a random factor inside a the contrast_range
    """

    def __init__(self, random_state, contrast_range=(0.2, 2)):
        assert (len(contrast_range) == 2)
        self.min_factor = contrast_range[0]
        self.max_factor = contrast_range[1]
        self.random_state = random_state

    def __call__(self, m):
        factor = self.random_state.uniform(self.min_factor, self.max_factor)

        if m.ndim==3:
            mean_intensity = np.mean(m)# take the mean intesity of the entire patch
            
            img_contrast = np.clip(mean_intensity + factor * (m - mean_intensity), 0, 1)
        elif m.ndim==4:
            mean_channels = np.mean(m, axis=(1, 2, 3)) # compute per channel mean intensity
            mean_channels = np.expand_dims(mean_channels, axis=1)
            mean_channels = np.expand_dims(mean_channels, axis=1)
            mean_channels = np.expand_dims(mean_channels, axis=1)

            img_contrast = np.clip(mean_channels + factor * (m - mean_channels), 0, 1)

        return img_contrast


class RandomBrightness(object):
    """
        Adjust the brightness of an image by a random factor inside a the brigtess_range
        Brightness range: tuple,float. If it's a tuple  the ranfom factor will be taken from (brighness_range[0],brightness_range[1])
        If it's float then the random factor will be taken from (-brighness_range,brightness_range).
        The intervals must be included in [-1,1]. If not, they would be clipped to [-1,1]
    """

    def __init__(self, random_state, brightness_range=0.2):
        if isinstance(brightness_range, tuple):
            assert (len(brightness_range) == 2)
            self.brightness_min = max(min(brightness_range[0], 1.0), -1.0)
            self.brightness_max = max(min(brightness_range[1], 1.0), -1.0)
        else:
            self.brightness_min = self.value = max(min(brightness_range, 1.0), -1.0)
            self.brightness_max = max(min(-brightness_range, 1.0), -1.0)
        self.random_state = random_state

    def __call__(self, m):
        brightness = self.random_state.uniform(-self.brightness_min, self.brightness_max)
        img_brightness = np.clip(m + brightness, 0, 1)

        return img_brightness

class RandomBrightnessContrast(object):
    """
        Adjust the brightness of an image by a random factor inside a the brigtess_range
        Brightness range: tuple,float. If it's a tuple  the ranfom factor will be taken from (brighness_range[0],brightness_range[1])
        If it's float then the random factor will be taken from (-brighness_range,brightness_range).
        The intervals must be included in [-1,1]. If not, they would be clipped to [-1,1]
    """

    def __init__(self, random_state,brightness_range=None, contrast_range=None):
        if contrast_range!=None:
            self.rand_contrast=RandomContrast(random_state, contrast_range)
        else:
            self.rand_contrast=RandomContrast(random_state)
        if brightness_range!=None:
            self.rand_brightness=RandomBrightness(random_state, brightness_range)
        else:
            self.rand_brightness=RandomBrightness(random_state)
        self.random_state = random_state

    def __call__(self, m):
        if self.random_state.uniform<0.5:#Alternates order of the Brightness and Contrast transforms
            img_transformed = self.rand_brightness(m)
            img_transformed = self.rand_contrast(img_transformed)
        else:
            img_transformed = self.rand_contrast(m)
            img_transformed = self.rand_brightness(img_transformed)
            

        return img_transformed

class AbstractLabelToBoundary:
    AXES = {
        0: (0, 1, 2),
        1: (0, 2, 1),
        2: (2, 0, 1)
    }

    def __init__(self, axes=(0, 1, 2), ignore_index=None):
        """
        :param axes: axes across which the boundary will be computed
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        """
        if isinstance(axes, int):
            assert axes in [0, 1, 2], "Axis must be one of [0, 1, 2]"
            axes = [self.AXES[axes]]
        elif isinstance(axes, list) or isinstance(axes, tuple):
            assert all(a in [0, 1, 2] for a in axes), "Axis must be one of [0, 1, 2]"
            assert len(set(axes)) == len(axes), "'axes' must be unique"
            axes = [self.AXES[a] for a in axes]
        else:
            raise ValueError(f"Unsupported 'axes' type {type(axes)}")
        self.axes = axes
        self.ignore_index = ignore_index

    def __call__(self, m):
        """
        Extract boundaries from a given 3D label tensor.
        :param m: input 3D tensor
        :return: binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """
        assert m.ndim == 3

        kernels = self.get_kernels()

        channels = [self._recover_ignore_index(np.abs(convolve(m, kernel)), m) for kernel in kernels]

        if len(channels) > 1:
            # stack if more than one channel
            result = np.stack(channels, axis=0)
        else:
            # otherwise just take first channel
            result = channels[0]

        # binarize the result
        result[result > 0] = 1
        return result

    @staticmethod
    def create_kernel(axis, offset):
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, 1, k_size), dtype=np.int)
        k[0, 0, 0] = 1
        k[0, 0, offset] = -1
        return np.transpose(k, axis)

    def _recover_ignore_index(self, input, orig):
        if self.ignore_index is not None:
            mask = orig == self.ignore_index
            input[mask] = self.ignore_index

        return input

    def get_kernels(self):
        raise NotImplementedError


class RandomLabelToBoundary(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the max_offset (thickness) of the border as well as the axes (direction) across which the boundary
    will be computed via the convolution operator. The axes and offset is picked at random every time you call
    the transformer (offset is picked form the range 1:max_offset), in order to make the network more robust against
    various thickness of borders in the ground truth (think of it as a boundary denoising scheme).
    """

    def __init__(self, max_offset=4, axes=(0, 1, 2), ignore_index=None):
        """
        :param max_offset: maximum offset in a given direction; in the runtime the offset will be randomly chosen
            from [1:max_offset] so that the network gets more resilient to the noise in the labels
        :param axes: axes across which the boundary will be computed
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        """
        super().__init__(axes, ignore_index)
        self.offsets = tuple(range(1, max_offset + 1))

    def get_kernels(self):
        axis = random.choice(self.axes)
        offset = random.choice(self.offsets)
        return [self.create_kernel(axis, offset)]


class LabelToBoundary(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the offsets (thickness) of the border as well as the axes (direction) across which the boundary
    will be computed via the convolution operator. The convolved images are stacked across the channel dimension (CxDxHxW)
    """

    def __init__(self, offsets, axes=(0, 1, 2), ignore_index=None):
        super().__init__(axes, ignore_index)
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
        for axis in self.axes:
            for offset in offsets:
                self.kernels.append(self.create_kernel(axis, offset))

    def get_kernels(self):
        return self.kernels


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

    def weight_transform(self):
        return Compose([
            ToTensor(expand_dims=False)
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

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


class Contrast_Brightness_StandardTransformer(BaseTransformer):
    """
    Standard data augmentation: random adjust of Contrast and Brightness + random flips across randomly picked axis + random 90 degrees rotations.
    """

    def __init__(self, mean, std, phase, label_dtype, contrast_range=(0.2, 2), brightness_range=0.2, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert (len(contrast_range) == 2)
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([RandomBrightnessContrast(np.random.RandomState(self.seed),self.brightness_range,self.contrast_range),
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

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


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

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


class Contrast_Brightness_IsotropicRotationTransformer(BaseTransformer):
    """
    Data augmentation to be used with isotropic 3D volumes: random adjust of Contrast and Brightness + random flips across randomly picked axis + random 90 deg
    rotations + random angle rotations across randomly picked axis.
    """

    def __init__(self, mean, std, phase, label_dtype, contrast_range=(0.2, 2), brightness_range=0.2, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert (len(contrast_range) == 2)
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([RandomBrightnessContrast(np.random.RandomState(self.seed),self.brightness_range,self.contrast_range),
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
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


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
                # rotate in XY only (ZYX axis order is assumed)
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
                # rotate in XY only (ZYX axis order is assumed)
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


class Contrast_Brightness_AnisotropicRotationTransformer(BaseTransformer):
    """
    Data augmentation to be used with anisotropic 3D volumes: random flips across randomly picked axis + random 90 deg
    rotations + random angle rotations across (1,0) axis.
    """

    def __init__(self, mean, std, phase, label_dtype, contrast_range=(0.2, 2), brightness_range=0.2, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert (len(contrast_range) == 2)
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([RandomBrightnessContrast(np.random.RandomState(self.seed),self.brightness_range,self.contrast_range),
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
                # rotate in XY only (ZYX axis order is assumed)
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return super().label_transform()

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)]),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


# Make sure to use mode='reflect' when doing RandomRotate otherwise the transform will produce unwanted boundary signal
class LabelToBoundaryTransformer(BaseTransformer):
    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']
        if 'ignore_index' in kwargs:
            self.ignore_index = kwargs['ignore_index']
        else:
            self.ignore_index = None

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                # this will give us 6 output channels with boundary signal
                LabelToBoundary(axes=(0, 1, 2), offsets=(1, 4), ignore_index=self.ignore_index),
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])
        else:
            return Compose([
                LabelToBoundary(axes=(0, 1, 2), offsets=(1, 4), ignore_index=self.ignore_index),
                # this will give us 6 output channels with boundary signal
                ToTensor(expand_dims=False, dtype=self.label_dtype)
            ])

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


# Make sure to use mode='reflect' when doing RandomRotate otherwise the transform will produce unwanted boundary signal
class RandomLabelToBoundaryTransformer(BaseTransformer):
    def __init__(self, mean, std, phase, label_dtype, **kwargs):
        super().__init__(mean=mean, std=std, phase=phase, label_dtype=label_dtype)
        assert 'angle_spectrum' in kwargs, "'angle_spectrum' argument required"
        self.angle_spectrum = kwargs['angle_spectrum']
        if 'ignore_index' in kwargs:
            self.ignore_index = kwargs['ignore_index']
        else:
            self.ignore_index = None

    def raw_transform(self):
        if self.phase == 'train':
            return Compose([
                Normalize(self.mean, self.std),
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ToTensor(expand_dims=True)
            ])
        else:
            return super().raw_transform()

    def label_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                # rotate in XY only and make sure mode='reflect' is used in order to prevent boundary artifacts
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                RandomLabelToBoundary(ignore_index=self.ignore_index),
                ToTensor(expand_dims=True, dtype=self.label_dtype)
            ])
        else:
            return Compose([
                RandomLabelToBoundary(ignore_index=self.ignore_index),
                # this will give us 6 output channels with boundary signal
                ToTensor(expand_dims=True, dtype=self.label_dtype)
            ])

    def weight_transform(self):
        if self.phase == 'train':
            return Compose([
                RandomFlip(np.random.RandomState(self.seed)),
                RandomRotate90(np.random.RandomState(self.seed)),
                RandomRotate(np.random.RandomState(self.seed), angle_spectrum=self.angle_spectrum, axes=[(2, 1)],
                             mode='reflect'),
                ToTensor(expand_dims=False)
            ])
        else:
            return super().weight_transform()


