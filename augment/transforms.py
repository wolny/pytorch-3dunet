import importlib

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.segmentation import find_boundaries
from skimage.filters import gaussian
from torchvision.transforms import Compose


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, **kwargs):
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

    def __init__(self, random_state, **kwargs):
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

    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='constant', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast:
    """
       Adjust the contrast of an image by a random factor inside a the contrast_range
    """

    def __init__(self, random_state, contrast_range=(0.25, 0.75), execution_probability=0.2, **kwargs):
        assert len(contrast_range) == 2
        self.min_factor, self.max_factor = contrast_range
        self.random_state = random_state
        self.execution_probability = execution_probability

    def __call__(self, m):
        factor = self.random_state.uniform(self.min_factor, self.max_factor)

        if self.random_state.uniform() < self.execution_probability:
            if m.ndim == 3:
                # take the mean intensity of the entire patch
                mean_intensity = np.mean(m)
            else:
                # if 4D then compute per channel mean intensity (assuming: CZYX axis order)
                mean_intensity = np.mean(m, axis=(1, 2, 3))
            return np.clip(mean_intensity + factor * (m - mean_intensity), 0, 1)

        return m


class RandomBrightness:
    """
        Adjust the brightness of an image by a random factor inside a the brightness_range
        Brightness range: tuple,float. If it's a tuple a random factor will be taken from (brightness_range[0], brightness_range[1])
        If it's float then the random factor will be taken from (-brightness_range,brightness_range).
        The intervals must be included in [-1,1]. If not, they would be clipped to [-1,1]
    """

    def __init__(self, random_state, brightness_range=0.1, **kwargs):
        if isinstance(brightness_range, tuple):
            assert len(brightness_range) == 2
            self.brightness_min, self.brightness_max = np.clip(brightness_range, -1., 1.)
        else:
            self.brightness_min, self.brightness_max = np.clip([-brightness_range, brightness_range], -1., 1.)
        self.random_state = random_state

    def __call__(self, m):
        brightness = self.random_state.uniform(self.brightness_min, self.brightness_max)
        return np.clip(m + brightness, 0, 1)


class RandomBrightnessContrast:
    """
        Apply RandomBrightness and RandomContrast interchangeably
    """

    def __init__(self, random_state, brightness_range=0.1, contrast_range=(0.25, 0.75), **kwargs):
        self.rand_contrast = RandomContrast(random_state, contrast_range)
        self.rand_brightness = RandomBrightness(random_state, brightness_range)
        self.random_state = random_state

    def __call__(self, m):
        if self.random_state.uniform() < 0.5:  # Alternates order of the Brightness and Contrast transforms
            m = self.rand_brightness(m)
            return self.rand_contrast(m)
        else:
            m = self.rand_contrast(m)
            return self.rand_brightness(m)


# it's relatively slow, i.e. ~1s per patch of size 64x200x200
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order!
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=15, sigma=3, **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smothing factor for Gaussian filter
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, m):
        assert m.ndim == 3
        dz = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dx = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha

        z_dim, y_dim, x_dim = m.shape
        z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
        indices = z + dz, y + dy, x + dx
        return map_coordinates(m, indices, order=self.spline_order, mode='reflect')


class AbstractLabelToBoundary:
    AXES_TRANSPOSE = [
        (0, 1, 2),  # X
        (0, 2, 1),  # Y
        (2, 0, 1)  # Z
    ]

    def __init__(self, ignore_index=None, aggregate_affinities=False, append_label=False, **kwargs):
        """
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        :param aggregate_affinities: aggregate affinities with the same offset across Z,Y,X axes
        :param append_label: if True append the orignal ground truth labels to the last channel
        """
        self.ignore_index = ignore_index
        self.aggregate_affinities = aggregate_affinities
        self.append_label = append_label

    def __call__(self, m):
        """
        Extract boundaries from a given 3D label tensor.
        :param m: input 3D tensor
        :return: binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """
        assert m.ndim == 3

        kernels = self.get_kernels()
        channels = np.stack([np.where(np.abs(convolve(m, kernel)) > 0, 1, 0) for kernel in kernels])
        results = []
        if self.aggregate_affinities:
            assert len(kernels) % 3 == 0, "Number of kernels must be divided by 3 (one kernel per offset per Z,Y,X axes"
            # aggregate affinities with the same offset
            for i in range(0, len(kernels), 3):
                # merge across X,Y,Z axes (logical OR)
                xyz_aggregated_affinities = np.logical_or.reduce(channels[i:i + 3, ...]).astype(np.int)
                # recover ignore index
                xyz_aggregated_affinities = _recover_ignore_index(xyz_aggregated_affinities, m, self.ignore_index)
                results.append(xyz_aggregated_affinities)
        else:
            results = [_recover_ignore_index(channels[i], m, self.ignore_index) for i in range(channels.shape[0])]

        if self.append_label:
            # append original input data
            results.append(m)

        # stack across channel dim
        return np.stack(results, axis=0)

    @staticmethod
    def create_kernel(axis, offset):
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, 1, k_size), dtype=np.int)
        k[0, 0, 0] = 1
        k[0, 0, offset] = -1
        return np.transpose(k, axis)

    def get_kernels(self):
        raise NotImplementedError


class StandardLabelToBoundary:
    def __init__(self, ignore_index=None, append_label=False, blur=False, **kwargs):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.blur = blur

    def __call__(self, m):
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2)
        if self.blur:
            boundaries = gaussian(boundaries, sigma=1)
            boundaries[boundaries >= 0.5] = 1
            boundaries[boundaries < 0.5] = 0

        results = [_recover_ignore_index(boundaries, m, self.ignore_index)]

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class RandomLabelToBoundary(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the max_offset (thickness) of the border. Then the offset is picked at random every time you call
    the transformer (offset is picked form the range 1:max_offset) for each axis and the boundary computed.
    One may use this scheme in order to make the network more robust against various thickness of borders in the ground
    truth  (think of it as a boundary denoising scheme).
    """

    def __init__(self, random_state, max_offset=8, ignore_index=None, append_label=False, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label, aggregate_affinities=False)
        self.random_state = random_state
        self.offsets = tuple(range(1, max_offset + 1))

    def get_kernels(self):
        rand_offset = self.random_state.choice(self.offsets)
        axis_ind = self.random_state.randint(3)
        rand_axis = self.AXES_TRANSPOSE[axis_ind]
        # return a single kernel
        return [self.create_kernel(rand_axis, rand_offset)]


class LabelToBoundary(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """

    def __init__(self, offsets, ignore_index=None, append_label=False, aggregate_affinities=False, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label,
                         aggregate_affinities=aggregate_affinities)
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
        for offset in offsets:
            for axis in self.AXES_TRANSPOSE:
                # create kernels for a given offset in every direction
                self.kernels.append(self.create_kernel(axis, offset))

    def get_kernels(self):
        return self.kernels


class Normalize:
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean, std, eps=1e-4, **kwargs):
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

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
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


def get_transformer(config, mean, std, phase):
    if phase == 'val':
        phase = 'test'

    assert phase in config, f'Cannot find transformer config for phase: {phase}'
    phase_config = config[phase]
    return Transformer(phase_config, mean, std)


class Transformer:
    def __init__(self, phase_config, mean, std):
        self.phase_config = phase_config
        self.config_base = {'mean': mean, 'std': std}
        self.seed = 47

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    def weight_transform(self):
        return self._create_transform('weight')

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input
