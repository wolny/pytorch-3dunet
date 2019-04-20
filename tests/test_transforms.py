import numpy as np

from augment.transforms import RandomLabelToAffinities, LabelToAffinities, Transformer


class TestTransforms:
    config = {'dtype': 'long'}

    def test_random_label_to_boundary(self):
        size = 20
        label = _diagonal_label_volume(size)

        transform = RandomLabelToAffinities(np.random.RandomState())
        result = transform(label)
        assert result.shape == (1,) + label.shape

    def test_random_label_to_boundary_with_ignore(self):
        size = 20
        label = _diagonal_label_volume(size, init=-1)

        transform = RandomLabelToAffinities(np.random.RandomState(), ignore_index=-1)
        result = transform(label)
        assert result.shape == (1,) + label.shape
        assert -1 in np.unique(result)

    def test_label_to_boundary(self):
        size = 20
        label = _diagonal_label_volume(size)

        # this transform will produce 2 channels
        transform = LabelToAffinities(offsets=(2, 4), aggregate_affinities=True)
        result = transform(label)
        assert result.shape == (2,) + label.shape
        assert np.array_equal(np.unique(result), [0, 1])

    def test_label_to_boundary_with_ignore(self):
        size = 20
        label = _diagonal_label_volume(size, init=-1)

        transform = LabelToAffinities(offsets=(2, 4), ignore_index=-1, aggregate_affinities=True)
        result = transform(label)
        assert result.shape == (2,) + label.shape
        assert np.array_equal(np.unique(result), [-1, 0, 1])

    def test_label_to_boundary_no_aggregate(self):
        size = 20
        label = _diagonal_label_volume(size)

        # this transform will produce 6 channels
        transform = LabelToAffinities(offsets=(2, 4), aggregate_affinities=False)
        result = transform(label)
        assert result.shape == (6,) + label.shape
        assert np.array_equal(np.unique(result), [0, 1])

    def test_BaseTransformer(self):
        config = {
            'raw': [{'name': 'Normalize'}, {'name': 'ToTensor', 'expand_dims': True}],
            'label': [{'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}],
            'weight': [{'name': 'ToTensor', 'expand_dims': False}]
        }
        transformer = Transformer(config, 0, 1)
        raw_transforms = transformer.raw_transform().transforms
        assert raw_transforms[0].mean == 0
        assert raw_transforms[0].std == 1
        assert raw_transforms[1].expand_dims
        label_transforms = transformer.label_transform().transforms
        assert not label_transforms[0].expand_dims
        assert label_transforms[0].dtype == 'long'
        weight_transforms = transformer.weight_transform().transforms
        assert not weight_transforms[0].expand_dims

    def test_StandardTransformer(self):
        config = {
            'raw': [
                {'name': 'Normalize'},
                {'name': 'RandomContrast', 'execution_probability': 0.5},
                {'name': 'RandomFlip'},
                {'name': 'RandomRotate90'},
                {'name': 'ToTensor', 'expand_dims': True}
            ],
            'label': [
                {'name': 'RandomFlip'},
                {'name': 'RandomRotate90'},
                {'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}
            ]
        }
        transformer = Transformer(config, 0, 1)
        raw_transforms = transformer.raw_transform().transforms
        assert raw_transforms[0].mean == 0
        assert raw_transforms[0].std == 1
        assert raw_transforms[1].execution_probability == 0.5
        assert raw_transforms[4].expand_dims
        label_transforms = transformer.label_transform().transforms
        assert len(label_transforms) == 3

    def test_AnisotropicRotationTransformer(self):
        config = {
            'raw': [
                {'name': 'Normalize'},
                {'name': 'RandomContrast', 'execution_probability': 0.5},
                {'name': 'RandomFlip'},
                {'name': 'RandomRotate90'},
                {'name': 'RandomRotate', 'angle_spectrum': 17, 'axes': [[2, 1]]},
                {'name': 'ToTensor', 'expand_dims': True}
            ],
            'label': [
                {'name': 'RandomFlip'},
                {'name': 'RandomRotate90'},
                {'name': 'RandomRotate', 'angle_spectrum': 17, 'axes': [[2, 1]]},
                {'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}
            ]
        }
        transformer = Transformer(config, 0, 1)
        raw_transforms = transformer.raw_transform().transforms
        assert raw_transforms[0].mean == 0
        assert raw_transforms[0].std == 1
        assert raw_transforms[1].execution_probability == 0.5
        assert raw_transforms[4].angle_spectrum == 17
        assert raw_transforms[4].axes == [[2, 1]]
        label_transforms = transformer.label_transform().transforms
        assert len(label_transforms) == 4

    def test_LabelToBoundaryTransformer(self):
        config = {
            'raw': [
                {'name': 'Normalize'},
                {'name': 'RandomContrast', 'execution_probability': 0.5},
                {'name': 'RandomFlip'},
                {'name': 'RandomRotate90'},
                {'name': 'RandomRotate', 'angle_spectrum': 17, 'axes': [[2, 1]], 'mode': 'reflect'},
                {'name': 'ToTensor', 'expand_dims': True}
            ],
            'label': [
                {'name': 'RandomFlip'},
                {'name': 'RandomRotate90'},
                {'name': 'RandomRotate', 'angle_spectrum': 17, 'axes': [[2, 1]], 'mode': 'reflect'},
                {'name': 'LabelToAffinities', 'offsets': [2, 4, 6, 8]},
                {'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}
            ]
        }
        transformer = Transformer(config, 0, 1)
        raw_transforms = transformer.raw_transform().transforms
        assert raw_transforms[0].mean == 0
        assert raw_transforms[0].std == 1
        assert raw_transforms[1].execution_probability == 0.5
        assert raw_transforms[4].angle_spectrum == 17
        assert raw_transforms[4].axes == [[2, 1]]
        assert raw_transforms[4].mode == 'reflect'
        label_transforms = transformer.label_transform().transforms
        assert label_transforms[2].angle_spectrum == 17
        assert label_transforms[2].axes == [[2, 1]]
        assert label_transforms[2].mode == 'reflect'
        # 3 conv kernels per offset
        assert len(label_transforms[3].kernels) == 12

    def test_RandomLabelToBoundaryTransformer(self):
        config = {
            'raw': [
                {'name': 'Normalize'},
                {'name': 'RandomContrast', 'execution_probability': 0.5},
                {'name': 'RandomFlip'},
                {'name': 'RandomRotate90'},
                {'name': 'RandomRotate', 'angle_spectrum': 17, 'axes': [[2, 1]], 'mode': 'reflect'},
                {'name': 'ToTensor', 'expand_dims': True}
            ],
            'label': [
                {'name': 'RandomFlip'},
                {'name': 'RandomRotate90'},
                {'name': 'RandomRotate', 'angle_spectrum': 17, 'axes': [[2, 1]], 'mode': 'reflect'},
                {'name': 'RandomLabelToAffinities', 'max_offset': 4},
                {'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}
            ]
        }
        transformer = Transformer(config, 0, 1)
        label_transforms = transformer.label_transform().transforms
        assert label_transforms[3].offsets == (1, 2, 3, 4)


def _diagonal_label_volume(size, init=1):
    label = init * np.ones((size, size, size), dtype=np.int)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if i + j > 2 * k:
                    label[i, j, k] = 3
    return label
