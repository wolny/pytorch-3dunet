from tempfile import NamedTemporaryFile

import h5py
import numpy as np

from datasets.hdf5 import HDF5Dataset


class TestHDF5Dataset:
    def test_hdf5_dataset(self):
        path = create_random_dataset((128, 128, 128))

        patch_shapes = [(127, 127, 127), (69, 70, 70), (32, 64, 64)]
        stride_shapes = [(1, 1, 1), (17, 23, 23), (32, 64, 64)]

        for patch_shape, stride_shape in zip(patch_shapes, stride_shapes):
            with h5py.File(path, 'r') as f:
                raw = f['raw'][...]
                label = f['label'][...]

                dataset = HDF5Dataset(path, patch_shape, stride_shape, phase='test',
                                      transformer_config=transformer_config,
                                      raw_internal_path='raw', label_internal_path='label')

                # create zero-arrays of the same shape as the original dataset in order to verify if every element
                # was visited during the iteration
                visit_raw = np.zeros_like(raw)
                visit_label = np.zeros_like(label)

                for (_, idx) in dataset:
                    visit_raw[idx] = 1
                    visit_label[idx] = 1

                # verify that every element was visited at least once
                assert np.all(visit_raw)
                assert np.all(visit_label)

    def test_hdf5_with_multiple_label_datasets(self):
        path = create_random_dataset((128, 128, 128), label_datasets=['label1', 'label2'])
        patch_shape = (32, 64, 64)
        stride_shape = (32, 64, 64)

        dataset = HDF5Dataset(path, patch_shape, stride_shape, phase='train', transformer_config=transformer_config,
                              raw_internal_path='raw', label_internal_path=['label1', 'label2'])

        for raw, labels in dataset:
            assert len(labels) == 2

    def test_hdf5_with_multiple_raw_and_label_datasets(self):
        path = create_random_dataset((128, 128, 128), raw_datasets=['raw1', 'raw2'],
                                     label_datasets=['label1', 'label2'])
        patch_shape = (32, 64, 64)
        stride_shape = (32, 64, 64)

        dataset = HDF5Dataset(path, patch_shape, stride_shape, phase='train', transformer_config=transformer_config,
                              raw_internal_path=['raw1', 'raw2'], label_internal_path=['label1', 'label2'])

        for raws, labels in dataset:
            assert len(raws) == 2
            assert len(labels) == 2

    def test_augmentation(self):
        raw = np.random.rand(32, 96, 96)
        label = np.zeros((3, 32, 96, 96))
        # assign raw to label's channels for ease of comparison
        for i in range(label.shape[0]):
            label[i] = raw

        tmp_file = NamedTemporaryFile()
        tmp_path = tmp_file.name
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('raw', data=raw)
            f.create_dataset('label', data=label)

        dataset = HDF5Dataset(tmp_path, patch_shape=(16, 64, 64), stride_shape=(8, 32, 32), phase='train',
                              transformer_config=transformer_config)

        for (img, label) in dataset:
            for i in range(label.shape[0]):
                assert np.allclose(img, label[i])


def create_random_dataset(shape, ignore_index=False, raw_datasets=['raw'], label_datasets=['label']):
    tmp_file = NamedTemporaryFile(delete=False)

    with h5py.File(tmp_file.name, 'w') as f:
        for raw_dataset in raw_datasets:
            f.create_dataset(raw_dataset, data=np.random.rand(*shape))

        for label_dataset in label_datasets:
            if ignore_index:
                f.create_dataset(label_dataset, data=np.random.randint(-1, 2, shape))
            else:
                f.create_dataset(label_dataset, data=np.random.randint(0, 2, shape))

    return tmp_file.name


transformer_config = {
    'train': {
        'raw': [
            {'name': 'RandomFlip'},
            {'name': 'RandomRotate90'},
            {'name': 'RandomRotate', 'angle_spectrum': 30, 'axes': [[1, 0]]},
            {'name': 'RandomRotate', 'angle_spectrum': 5, 'axes': [[2, 1]]},
            {'name': 'ToTensor', 'expand_dims': True}
        ],
        'label': [
            {'name': 'RandomFlip'},
            {'name': 'RandomRotate90'},
            {'name': 'RandomRotate', 'angle_spectrum': 30, 'axes': [[1, 0]]},
            {'name': 'RandomRotate', 'angle_spectrum': 5, 'axes': [[2, 1]]},
            {'name': 'ToTensor', 'expand_dims': False}
        ]
    },
    'test': {
        'raw': [{'name': 'ToTensor', 'expand_dims': True}],
        'label': [{'name': 'ToTensor', 'expand_dims': False}]
    }
}
