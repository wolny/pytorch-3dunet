import numpy as np
import pytest

from augment.transforms import TransformerBuilder, RandomLabelToBoundary, LabelToBoundary, BaseTransformer, \
    StandardTransformer, IsotropicRotationTransformer, AnisotropicRotationTransformer, LabelToBoundaryTransformer, \
    RandomLabelToBoundaryTransformer


class TestTransforms:
    config = {'label_dtype': 'long'}

    def test_random_label_to_boundary(self):
        size = 20
        label = _diagonal_label_volume(size)

        transform = RandomLabelToBoundary()
        result = transform(label)
        assert result.shape == label.shape

    def test_random_label_to_boundary_with_ignore(self):
        size = 20
        label = _diagonal_label_volume(size, init=-1)

        transform = RandomLabelToBoundary(ignore_index=-1)
        result = transform(label)
        assert result.shape == label.shape
        assert -1 in np.unique(result)

    def test_label_to_boundary(self):
        size = 20
        label = _diagonal_label_volume(size)

        # this transform will produce 3 channels
        transform = LabelToBoundary(offsets=2)
        result = transform(label)
        assert result.shape == (3,) + label.shape
        assert np.array_equal(np.unique(result), [0, 1])

    def test_label_to_boundary_with_ignore(self):
        size = 20
        label = _diagonal_label_volume(size, init=-1)

        transform = LabelToBoundary(offsets=2, ignore_index=-1)
        result = transform(label)
        assert result.shape == (3,) + label.shape
        assert np.array_equal(np.unique(result), [-1, 0, 1])

    def test_transformer_builder_build(self):
        builder = TransformerBuilder(BaseTransformer, self.config)
        with pytest.raises(AssertionError):
            builder.build()

    def test_BaseTransformer(self):
        builder = TransformerBuilder(BaseTransformer, self.config)
        builder.mean = 0
        builder.std = 1
        builder.phase = 'train'
        builder.build()

    def test_StandardTransformer(self):
        builder = TransformerBuilder(StandardTransformer, self.config)
        builder.mean = 0
        builder.std = 1
        builder.phase = 'train'
        builder.build()

    def test_IsotropicRotationTransformer(self):
        config = self.config.copy()
        config['angle_spectrum'] = 17
        builder = TransformerBuilder(IsotropicRotationTransformer, config)
        builder.mean = 0
        builder.std = 1
        builder.phase = 'train'

        transformer = builder.build()
        assert transformer.angle_spectrum == 17

    def test_AnisotropicRotationTransformer(self):
        config = self.config.copy()
        config['angle_spectrum'] = 17
        builder = TransformerBuilder(AnisotropicRotationTransformer, config)
        builder.mean = 0
        builder.std = 1
        builder.phase = 'train'

        transformer = builder.build()
        assert transformer.angle_spectrum == 17

    def test_LabelToBoundaryTransformer(self):
        config = self.config.copy()
        config['angle_spectrum'] = 17
        config['ignore_index'] = -1
        builder = TransformerBuilder(LabelToBoundaryTransformer, config)
        builder.mean = 0
        builder.std = 1
        builder.phase = 'train'

        transformer = builder.build()
        assert transformer.angle_spectrum == 17
        assert transformer.ignore_index == -1

    def test_RandomLabelToBoundaryTransformer(self):
        config = self.config.copy()
        config['angle_spectrum'] = 17
        config['ignore_index'] = -1
        builder = TransformerBuilder(RandomLabelToBoundaryTransformer, config)
        builder.mean = 0
        builder.std = 1
        builder.phase = 'train'

        transformer = builder.build()
        assert transformer.angle_spectrum == 17
        assert transformer.ignore_index == -1


def _diagonal_label_volume(size, init=1):
    label = init * np.ones((size, size, size), dtype=np.int)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if i + j > 2 * k:
                    label[i, j, k] = 3
    return label
