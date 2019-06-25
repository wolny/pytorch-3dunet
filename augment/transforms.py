import importlib

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
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
        Adjust the brightness of an image by a random factor.
    """

    def __init__(self, random_state, factor=0.5, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        self.factor = factor
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            brightness_factor = self.factor + self.random_state.uniform()
            return np.clip(m * brightness_factor, 0, 1)

        return m


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=3 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order!
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=15, sigma=3, execution_probability=0.3, **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim == 3
            dz = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
            dx = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha

            z_dim, y_dim, x_dim = m.shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx
            return map_coordinates(m, indices, order=self.spline_order, mode='reflect')

        return m


def blur_boundary(boundary, sigma):
    boundary = gaussian(boundary, sigma=sigma)
    boundary[boundary >= 0.5] = 1
    boundary[boundary < 0.5] = 0
    return boundary


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
        :param blur: Gaussian blur the boundaries
        :param sigma: standard deviation for Gaussian kernel
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
        boundary_arr = [np.where(np.abs(convolve(m, kernel)) > 0, 1, 0) for kernel in kernels]
        channels = np.stack(boundary_arr)
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
    def __init__(self, ignore_index=None, append_label=False, blur=False, sigma=1, **kwargs):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.blur = blur
        self.sigma = sigma

    def __call__(self, m):
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2)
        if self.blur:
            boundaries = blur_boundary(boundaries, self.sigma)

        results = [_recover_ignore_index(boundaries, m, self.ignore_index)]

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class RandomLabelToAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the max_offset (thickness) of the border. Then the offset is picked at random every time you call
    the transformer (offset is picked form the range 1:max_offset) for each axis and the boundary computed.
    One may use this scheme in order to make the network more robust against various thickness of borders in the ground
    truth  (think of it as a boundary denoising scheme).
    """

    def __init__(self, random_state, max_offset=10, ignore_index=None, append_label=False, z_offset_scale=2, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label, aggregate_affinities=False)
        self.random_state = random_state
        self.offsets = tuple(range(1, max_offset + 1))
        self.z_offset_scale = z_offset_scale

    def get_kernels(self):
        rand_offset = self.random_state.choice(self.offsets)
        axis_ind = self.random_state.randint(3)
        # scale down z-affinities due to anisotropy
        if axis_ind == 2:
            rand_offset = max(1, rand_offset // self.z_offset_scale)

        rand_axis = self.AXES_TRANSPOSE[axis_ind]
        # return a single kernel
        return [self.create_kernel(rand_axis, rand_offset)]


class LabelToAffinities(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels (which can be seen
    as an affinity graph: https://arxiv.org/pdf/1706.00120.pdf)
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """

    def __init__(self, offsets, ignore_index=None, append_label=False, aggregate_affinities=False, z_offsets=None,
                 **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label,
                         aggregate_affinities=aggregate_affinities)

        assert isinstance(offsets, list) or isinstance(offsets, tuple), 'offsets must be a list or a tuple'
        assert all(a > 0 for a in offsets), "'offsets must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"
        if z_offsets is not None:
            assert len(offsets) == len(z_offsets), 'z_offsets length must be the same as the length of offsets'
        else:
            # if z_offsets is None just use the offsets for z-affinities
            z_offsets = list(offsets)
        self.z_offsets = z_offsets

        self.kernels = []
        # create kernel for every axis-offset pair
        for xy_offset, z_offset in zip(offsets, z_offsets):
            for axis_ind, axis in enumerate(self.AXES_TRANSPOSE):
                final_offset = xy_offset
                if axis_ind == 2:
                    final_offset = z_offset
                # create kernels for a given offset in every direction
                self.kernels.append(self.create_kernel(axis, final_offset))

    def get_kernels(self):
        return self.kernels


class LabelToBoundaryAndAffinities:
    """
    Combines the StandardLabelToBoundary and LabelToAffinities in the hope
    that that training the network to predict both would improve the main task: boundary prediction.
    """

    def __init__(self, xy_offsets, z_offsets, append_label=False, blur=False, sigma=1, ignore_index=None, **kwargs):
        self.l2b = StandardLabelToBoundary(blur=blur, sigma=sigma, ignore_index=ignore_index)
        self.l2a = LabelToAffinities(offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label,
                                     ignore_index=ignore_index)

    def __call__(self, m):
        boundary = self.l2b(m)
        affinities = self.l2a(m)
        return np.concatenate((boundary, affinities), axis=0)


class LabelToMaskAndAffinities:
    def __init__(self, xy_offsets, z_offsets, append_label=False, background=0, ignore_index=None, **kwargs):
        self.background = background
        self.l2a = LabelToAffinities(offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label,
                                     ignore_index=ignore_index)

    def __call__(self, m):
        mask = m > self.background
        mask = np.expand_dims(mask.astype(np.uint8), axis=0)
        affinities = self.l2a(m)
        return np.concatenate((mask, affinities), axis=0)


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


class RangeNormalize:
    def __init__(self, max_value=255, **kwargs):
        self.max_value = max_value

    def __call__(self, m):
        return m / self.max_value


class GaussianNoise:
    def __init__(self, random_state, max_sigma, max_value=255, **kwargs):
        self.random_state = random_state
        self.max_sigma = max_sigma
        self.max_value = max_value

    def __call__(self, m):
        # pick std dev from [0; max_sigma]
        std = self.random_state.randint(self.max_sigma)
        gaussian_noise = self.random_state.normal(0, std, m.shape)
        noisy_m = m + gaussian_noise
        return np.clip(noisy_m, 0, self.max_value).astype(m.dtype)


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
