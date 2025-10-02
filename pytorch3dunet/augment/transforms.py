import importlib
import random
from collections.abc import Callable

import numpy as np
import torch
from scipy.ndimage import convolve, gaussian_filter, map_coordinates, rotate
from skimage import exposure, measure
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms: list[Callable[[np.ndarray], np.ndarray]]):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m


class RandomFlip:
    """Randomly flips the image across the given axes.

    Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    Args:
        random_state: Random state for reproducibility.
        axis_prob: Probability of flipping along each axis. Default: 0.5.
    """

    def __init__(self, random_state: np.random.RandomState, axis_prob: float = 0.5, **kwargs):
        assert random_state is not None, "RandomState cannot be None"
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m: np.ndarray) -> np.ndarray:
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class RandomRotate90:
    """Rotate an array by 90 degrees around a randomly chosen plane.

    Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: Assumes DHW axis order (that's why rotation is performed across (1,2) axis).

    Args:
        random_state: Random state for reproducibility.
    """

    def __init__(self, random_state: np.random.RandomState, **kwargs):
        self.random_state = random_state
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, m: np.ndarray) -> np.ndarray:
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomRotate:
    """Rotate an array by a random degrees from (-angle_spectrum, angle_spectrum) interval.

    Rotation axis is picked at random from the list of provided axes.

    Args:
        random_state: Random state for reproducibility.
        angle_spectrum: Maximum rotation angle. Default: 30.
        axes: List of rotation axes. Default: [(1, 0), (2, 1), (2, 0)].
        mode: Interpolation mode. Default: 'reflect'.
        order: Interpolation order. Default: 0.
    """

    def __init__(
        self,
        random_state: np.random.RandomState,
        angle_spectrum: int = 30,
        axes: list = None,
        mode: str = "reflect",
        order: int = 0,
        **kwargs,
    ):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m: np.ndarray) -> np.ndarray:
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [
                rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
                for c in range(m.shape[0])
            ]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast:
    """Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.

    Args:
        random_state: Random state for reproducibility.
        alpha: Range of contrast adjustment factor. Default: (0.5, 1.5).
        mean: Mean value for contrast adjustment. Default: 0.0.
        execution_probability: Probability of applying this transform. Default: 0.1.
    """

    def __init__(
        self,
        random_state: np.random.RandomState,
        alpha: tuple[float, float] = (0.5, 1.5),
        mean: float = 0.0,
        execution_probability: float = 0.1,
        **kwargs,
    ):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)

        return m


class RandomGammaCorrection:
    """Adjust contrast by scaling each voxel to `v ** gamma`.

    Args:
        random_state: Random state for reproducibility.
        gamma: Range of gamma values. Default: (0.5, 1.5).
        execution_probability: Probability of applying this transform. Default: 0.1.
    """

    def __init__(
        self,
        random_state: np.random.RandomState,
        gamma: tuple[float, float] = (0.5, 1.5),
        execution_probability: float = 0.1,
        **kwargs,
    ):
        self.random_state = random_state
        assert len(gamma) == 2
        self.gamma = gamma
        self.execution_probability = execution_probability

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.random_state.uniform() < self.execution_probability:
            # rescale intensity values to [0, 1]
            m = exposure.rescale_intensity(m, out_range=(0, 1))
            gamma = self.random_state.uniform(self.gamma[0], self.gamma[1])
            return exposure.adjust_gamma(m, gamma)

        return m


class ElasticDeformation:
    """Apply elastic deformations of 3D patches on a per-voxel mesh.
    This augmentation  relatively slow (~1s per patch of size 64x200x200), so use multiple workers in the DataLoader.
    Remember to use spline_order=0 when transforming the labels.

    Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62

    Args:
        random_state: Random state for reproducibility.
        spline_order: The order of spline interpolation (use 0 for labeled images).
        alpha: Scaling factor for deformations. Default: 2000.
        sigma: Smoothing factor for Gaussian filter. Default: 50.
        execution_probability: Probability of executing this transform. Default: 0.1.
        apply_3d: If True apply deformations in each axis. Default: True.
    """

    def __init__(
        self,
        random_state: np.random.RandomState,
        spline_order: int,
        alpha: int = 2000,
        sigma: int = 50,
        execution_probability: float = 0.1,
        apply_3d: bool = True,
        **kwargs,
    ):
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]

            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape

            if self.apply_3d:
                dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
            else:
                dz = np.zeros_like(m)

            dy, dx = [
                gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
                for _ in range(2)
            ]

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing="ij")
            indices = z + dz, y + dy, x + dx

            if m.ndim == 3:
                return map_coordinates(m, indices, order=self.spline_order, mode="reflect")
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode="reflect") for c in m]
                return np.stack(channels, axis=0)

        return m


class CropToFixed:
    """Crop or pad the input array to a fixed size.

    Args:
        random_state: Random state for reproducibility.
        size: Desired output size (y, x). Default: (256, 256).
        centered: If True, always crop/pad around the center. Default: False.
    """

    def __init__(
        self, random_state: np.random.RandomState, size: tuple[int, int] = (256, 256), centered: bool = False, **kwargs
    ):
        self.random_state = random_state
        self.crop_y, self.crop_x = size
        self.centered = centered

    def __call__(self, m: np.ndarray) -> np.ndarray:
        def _padding(pad_total: int) -> tuple[int, int]:
            half_total = pad_total // 2
            return half_total, pad_total - half_total

        def _rand_range_and_pad(crop_size: int, max_size: int) -> tuple[int, tuple[int, int]]:
            """
            Returns a tuple:
                max_value for the corner dimension. The corner dimension is chosen as `self.random_state(max_value)`
                pad: padding in both directions; if crop_size is lt max_size the pad is 0
            """
            if crop_size < max_size:
                return max_size - crop_size, (0, 0)
            else:
                return 1, _padding(crop_size - max_size)

        def _start_and_pad(crop_size: int, max_size: int) -> tuple[int, tuple[int, int]]:
            if crop_size < max_size:
                return (max_size - crop_size) // 2, (0, 0)
            else:
                return 0, _padding(crop_size - max_size)

        assert m.ndim in (3, 4)
        if m.ndim == 3:
            _, y, x = m.shape
        else:
            _, _, y, x = m.shape

        if not self.centered:
            y_range, y_pad = _rand_range_and_pad(self.crop_y, y)
            x_range, x_pad = _rand_range_and_pad(self.crop_x, x)

            y_start = self.random_state.randint(y_range)
            x_start = self.random_state.randint(x_range)

        else:
            y_start, y_pad = _start_and_pad(self.crop_y, y)
            x_start, x_pad = _start_and_pad(self.crop_x, x)

        if m.ndim == 3:
            result = m[:, y_start : y_start + self.crop_y, x_start : x_start + self.crop_x]
            return np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode="reflect")
        else:
            channels = []
            for c in range(m.shape[0]):
                result = m[c][:, y_start : y_start + self.crop_y, x_start : x_start + self.crop_x]
                channels.append(np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode="reflect"))
            return np.stack(channels, axis=0)


class StandardLabelToBoundary:
    """Converts a given volumetric label array to binary mask corresponding to borders between labels.

    Args:
        ignore_index: Label to ignore in the output.
        append_label: If True, stack the borders and original labels across channel dim. Default: False.
        mode: Boundary detection mode. Default: 'thick'.
        foreground: If True, include foreground mask (i.e everything greater than 0) in the first channel of the result.
            Default: False.
    """

    def __init__(
        self,
        ignore_index: int = None,
        append_label: bool = False,
        mode: str = "thick",
        foreground: bool = False,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m: np.ndarray) -> np.ndarray:
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        boundaries = boundaries.astype("int32")

        results = []
        if self.foreground:
            foreground = (m > 0).astype("uint8")
            results.append(_recover_ignore_index(foreground, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class BlobsToMask:
    """Returns binary mask from labeled image of blob like objects.

    Every label greater than 0 is treated as foreground.

    Args:
        append_label: If True, append original labels. Default: False.
        boundary: If True, compute boundaries. Default: False.
        cross_entropy: If True, use cross entropy format. Default: False.
    """

    def __init__(self, append_label: bool = False, boundary: bool = False, cross_entropy: bool = False, **kwargs):
        self.cross_entropy = cross_entropy
        self.boundary = boundary
        self.append_label = append_label

    def __call__(self, m):
        assert m.ndim == 3

        # get the segmentation mask
        mask = (m > 0).astype("uint8")
        results = [mask]

        if self.boundary:
            outer = find_boundaries(m, connectivity=2, mode="outer")
            if self.cross_entropy:
                # boundary is class 2
                mask[outer > 0] = 2
                results = [mask]
            else:
                results.append(outer)

        if self.append_label:
            results.append(m)

        return np.stack(results, axis=0)


class AbstractLabelToBoundary:
    """Abstract base class for label to boundary conversion.

    Args:
        ignore_index: Label to be ignored in the output, i.e. after computing the boundary the label
            ignore_index will be restored where it was in the patch originally.
        aggregate_affinities: Aggregate affinities with the same offset across Z,Y,X axes. Default: False.
        append_label: If True append the original ground truth labels to the last channel. Default: False.
    """

    AXES_TRANSPOSE = [
        (0, 1, 2),  # X
        (0, 2, 1),  # Y
        (2, 0, 1),  # Z
    ]

    def __init__(
        self, ignore_index: int = None, aggregate_affinities: bool = False, append_label: bool = False, **kwargs
    ):
        self.ignore_index = ignore_index
        self.aggregate_affinities = aggregate_affinities
        self.append_label = append_label

    def __call__(self, m: np.ndarray) -> np.ndarray:
        """Extract boundaries from a given 3D label tensor.

        Args:
            m: Input 3D tensor.

        Returns:
            Binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background.
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
                xyz_aggregated_affinities = np.logical_or.reduce(channels[i : i + 3, ...]).astype(np.int32)
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
    def create_kernel(axis: int | tuple, offset: int) -> np.ndarray:
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, 1, k_size), dtype=np.int32)
        k[0, 0, 0] = 1
        k[0, 0, offset] = -1
        return np.transpose(k, axis)

    def get_kernels(self):
        raise NotImplementedError


class RandomLabelToAffinities(AbstractLabelToBoundary):
    """Converts a given volumetric label array to binary mask corresponding to borders between labels.

    One specifies the max_offset (thickness) of the border. Then the offset is picked at random every time
    you call the transformer (offset is picked from the range 1:max_offset) for each axis and the boundary computed.
    One may use this scheme in order to make the network more robust against various thickness of borders in the
    ground truth (think of it as a boundary denoising scheme).

    Args:
        random_state: Random state for reproducibility.
        max_offset: Maximum offset for boundary thickness. Default: 10.
        ignore_index: Label to ignore in the output.
        append_label: If True, append original labels. Default: False.
        z_offset_scale: Scale factor for z-axis offsets. Default: 2.
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
    """Converts a given volumetric label array to binary mask corresponding to borders between labels.

    This can be seen as an affinity graph: https://arxiv.org/pdf/1706.00120.pdf
    One specifies the offsets (thickness) of the border. The boundary will be computed via the convolution operator.

    Args:
        offsets: List of offsets for boundary thickness.
        ignore_index: Label to ignore in the output.
        append_label: If True, append original labels. Default: False.
        aggregate_affinities: If True, aggregate affinities. Default: False.
        z_offsets: Offsets for z-axis, if different from xy offsets.
    """

    def __init__(
        self, offsets, ignore_index=None, append_label=False, aggregate_affinities=False, z_offsets=None, **kwargs
    ):
        super().__init__(
            ignore_index=ignore_index, append_label=append_label, aggregate_affinities=aggregate_affinities
        )

        assert isinstance(offsets, list) or isinstance(offsets, tuple), "offsets must be a list or a tuple"
        assert all(a > 0 for a in offsets), "'offsets' must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"
        if z_offsets is not None:
            assert len(offsets) == len(z_offsets), "z_offsets length must be the same as the length of offsets"
        else:
            # if z_offsets is None just use the offsets for z-affinities
            z_offsets = list(offsets)
        self.z_offsets = z_offsets

        self.kernels = []
        # create kernel for every axis-offset pair
        for xy_offset, z_offset in zip(offsets, z_offsets, strict=True):
            for axis_ind, axis in enumerate(self.AXES_TRANSPOSE):
                final_offset = xy_offset
                if axis_ind == 2:
                    final_offset = z_offset
                # create kernels for a given offset in every direction
                self.kernels.append(self.create_kernel(axis, final_offset))

    def get_kernels(self):
        return self.kernels


class LabelToZAffinities(AbstractLabelToBoundary):
    """Converts a given volumetric label array to binary mask corresponding to borders between labels in Z-axis only.

    This can be seen as an affinity graph: https://arxiv.org/pdf/1706.00120.pdf
    One specifies the offsets (thickness) of the border. The boundary will be computed via the convolution operator.

    Args:
        offsets: List of offsets for boundary thickness.
        ignore_index: Label to ignore in the output.
        append_label: If True, append original labels. Default: False.
    """

    def __init__(self, offsets, ignore_index=None, append_label=False, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label)

        assert isinstance(offsets, list) or isinstance(offsets, tuple), "offsets must be a list or a tuple"
        assert all(a > 0 for a in offsets), "'offsets must be positive"
        assert len(set(offsets)) == len(offsets), "'offsets' must be unique"

        self.kernels = []
        z_axis = self.AXES_TRANSPOSE[2]
        # create kernels
        for z_offset in offsets:
            self.kernels.append(self.create_kernel(z_axis, z_offset))

    def get_kernels(self):
        return self.kernels


class LabelToBoundaryAndAffinities:
    """Combines the StandardLabelToBoundary and LabelToAffinities.

    The hope is that training the network to predict both would improve the main task: boundary prediction.

    Args:
        xy_offsets: Offsets for XY axes.
        z_offsets: Offsets for Z axis.
        append_label: If True, append original labels. Default: False.
        ignore_index: Label to ignore in the output.
        mode: Boundary detection mode. Default: 'thick'.
        foreground: If True, include foreground mask. Default: False.
    """

    def __init__(
        self,
        xy_offsets: list,
        z_offsets: list,
        append_label: bool = False,
        ignore_index: int = None,
        mode: str = "thick",
        foreground: bool = False,
        **kwargs,
    ):
        # blur only StandardLabelToBoundary results; we don't want to blur the affinities
        self.l2b = StandardLabelToBoundary(ignore_index=ignore_index, mode=mode, foreground=foreground)
        self.l2a = LabelToAffinities(
            offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label, ignore_index=ignore_index
        )

    def __call__(self, m: np.ndarray) -> np.ndarray:
        boundary = self.l2b(m)
        affinities = self.l2a(m)
        return np.concatenate((boundary, affinities), axis=0)


class LabelToMaskAndAffinities:
    """
    Similar to LabelToBoundaryAndAffinities but instead of computing the boundary we just compute the foreground
    in the first channel (everything greater than background is foreground).
    """

    def __init__(self, xy_offsets, z_offsets, append_label=False, background=0, ignore_index=None, **kwargs):
        self.background = background
        self.l2a = LabelToAffinities(
            offsets=xy_offsets, z_offsets=z_offsets, append_label=append_label, ignore_index=ignore_index
        )

    def __call__(self, m: np.ndarray) -> np.ndarray:
        mask = m > self.background
        mask = np.expand_dims(mask.astype(np.uint8), axis=0)
        affinities = self.l2a(m)
        return np.concatenate((mask, affinities), axis=0)


class Standardize:
    """Apply Z-score normalization to a given input tensor.

    Re-scales the values to be 0-mean and 1-std.

    Args:
        eps: Small value to prevent division by zero. Default: 1e-10.
        mean: Pre-computed mean value.
        std: Pre-computed standard deviation value.
        channelwise: If True, normalize per-channel. Default: False.
    """

    def __init__(self, eps: float = 1e-10, mean: float = None, std: float = None, channelwise: bool = False, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


class PercentileNormalizer:
    """Apply percentile normalization to a given input tensor."""

    def __init__(self, pmin: float = 1.0, pmax: float = 99.6, channelwise: bool = False, eps: float = 1e-10, **kwargs):
        self.eps = eps
        self.pmin = pmin
        self.pmax = pmax
        self.channelwise = channelwise

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.channelwise:
            axes = list(range(m.ndim))
            # average across channels
            axes = tuple(axes[1:])
            pmin = np.percentile(m, self.pmin, axis=axes, keepdims=True)
            pmax = np.percentile(m, self.pmax, axis=axes, keepdims=True)
        else:
            pmin = np.percentile(m, self.pmin)
            pmax = np.percentile(m, self.pmax)

        return (m - pmin) / (pmax - pmin + self.eps)


class Normalize:
    """Apply simple min-max scaling to a given input tensor.

    Shrinks the range of the data to a fixed range of [-1, 1] or in case of norm01==True to [0, 1].

    Args:
        min_value: Minimum value for clipping. Default: None (use min of the input array).
        max_value: Maximum value for clipping. Default: None (use max of the input array).
        norm01: If True, normalize to [0, 1] instead of [-1, 1]. Default: False.
        eps: Small value to prevent division by zero. Default: 1e-10.
    """

    def __init__(
        self, min_value: float = None, max_value: float = None, norm01: bool = False, eps: float = 1e-10, **kwargs
    ):
        if min_value is not None and max_value is not None:
            assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.norm01 = norm01
        self.eps = eps

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.min_value is None:
            min_value = np.min(m)
        else:
            min_value = self.min_value

        if self.max_value is None:
            max_value = np.max(m)
        else:
            max_value = self.max_value

        # calculate norm_0_1 with min_value / max_value with the same dimension
        # in case of channelwise application
        norm_0_1 = (m - min_value) / (max_value - min_value + self.eps)

        if self.norm01:
            return np.clip(norm_0_1, 0, 1)
        else:
            return np.clip(2 * norm_0_1 - 1, -1, 1)


class AdditiveGaussianNoise:
    """Add Gaussian noise to a given input tensor."""

    def __init__(
        self,
        random_state: np.random.RandomState,
        scale: tuple[float, float] = (0.0, 1.0),
        execution_probability: float = 0.1,
        **kwargs,
    ):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.scale = scale

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            return m + gaussian_noise
        return m


class AdditivePoissonNoise:
    """Add Poisson noise to a given input tensor."""

    def __init__(
        self,
        random_state: np.random.RandomState,
        lam: tuple[float, float] = (0.0, 1.0),
        execution_probability: float = 0.1,
        **kwargs,
    ):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.lam = lam

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.random_state.uniform() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)
            return m + poisson_noise
        return m


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor.

    Args:
        expand_dims (bool): if True, adds a channel dimension to the input data
        dtype (np.dtype): the desired output data type
        normalize (bool): zero-one normalization of the input data
    """

    def __init__(self, expand_dims: bool, dtype: np.dtype = np.float32, normalize: bool = False, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype
        self.normalize = normalize

    def __call__(self, m: np.ndarray) -> torch.Tensor:
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        if self.normalize:
            # avoid division by zero
            m = (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-10)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Relabel:
    """Relabel a numpy array of labels into consecutive numbers.

    E.g. [10, 10, 0, 6, 6] -> [2, 2, 0, 1, 1]. Useful when one has an instance segmentation volume
    at hand and would like to create a one-hot-encoding for it. Without a consecutive labeling the task would be harder.

    Args:
        append_original: If True, append original labels. Default: False.
        run_cc: If True, run connected components. Default: True.
        ignore_label: Label to ignore.
    """

    def __init__(self, append_original: bool = False, run_cc: bool = True, ignore_label: int = None, **kwargs):
        self.append_original = append_original
        self.ignore_label = ignore_label
        self.run_cc = run_cc

        if ignore_label is not None:
            assert append_original, (
                "ignore_label present, so append_original must be true, so that one can localize the ignore region"
            )

    def __call__(self, m: np.ndarray) -> np.ndarray:
        orig = m
        if self.run_cc:
            # assign 0 to the ignore region
            m = measure.label(m, background=self.ignore_label)

        _, unique_labels = np.unique(m, return_inverse=True)
        result = unique_labels.reshape(m.shape)
        if self.append_original:
            result = np.stack([result, orig])
        return result


class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, m: np.ndarray) -> np.ndarray:
        return m


class RgbToLabel:
    """Convert a RGB image to a single channel label image."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = np.array(img)
        assert img.ndim == 3 and img.shape[2] == 3
        result = img[..., 0] * 65536 + img[..., 1] * 256 + img[..., 2]
        return result


class LabelToTensor:
    """Convert a given input numpy.ndarray label array into torch.Tensor of dtype int64."""

    def __call__(self, m: np.ndarray) -> torch.Tensor:
        m = np.array(m)
        return torch.from_numpy(m.astype(dtype="int64"))


class GaussianBlur3D:
    """Apply Gaussian blur to a given input tensor."""

    def __init__(self, sigma: tuple[float, float] = (0.1, 2.0), execution_probability: float = 0.5, **kwargs):
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if random.random() < self.execution_probability:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = gaussian(x, sigma=sigma)
            return x
        return x


class Transformer:
    """Factory class for creating data augmentation pipelines."""

    def __init__(self, phase_config: dict, base_config: dict):
        self.phase_config = phase_config
        self.config_base = base_config
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def raw_transform(self):
        return self._create_transform("raw")

    def label_transform(self):
        return self._create_transform("label")

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module("pytorch3dunet.augment.transforms")
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f"Could not find {name} transform"
        return Compose([self._create_augmentation(c) for c in self.phase_config[name]])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config["random_state"] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config["name"])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input
