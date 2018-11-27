from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from torchvision.transforms import Compose

import augment.transforms as transforms
from datasets.hdf5 import HDF5Dataset


class TestHDF5Dataset(object):
    def test_hdf5_dataset(self):
        path = create_random_dataset((128, 128, 128))

        patch_shapes = [(127, 127, 127), (69, 70, 70), (32, 64, 64)]
        stride_shapes = [(1, 1, 1), (17, 23, 23), (32, 64, 64)]

        for patch_shape, stride_shape in zip(patch_shapes, stride_shapes):
            with h5py.File(path, 'r') as f:
                raw = f['raw'][...]
                label = f['label'][...]

                dataset = HDF5Dataset(path, patch_shape, stride_shape, 'test')

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

    def test_augmentation(self):
        raw = np.random.rand(32, 96, 96)
        label = np.zeros((3, 32, 96, 96))
        # assign raw to label's channels for ease of comparison
        for i in range(label.shape[0]):
            label[i] = raw

        tmp_file = NamedTemporaryFile()
        tmp_path = tmp_file.name
        f = h5py.File(tmp_path, 'w')

        f.create_dataset('raw', data=raw)
        f.create_dataset('label', data=label)
        f.close()

        dataset = CustomAugmentation(tmp_path, patch_shape=(16, 64, 64), stride_shape=(1, 1, 1), phase='train')

        for (img, label) in dataset:
            for i in range(label.shape[0]):
                assert np.allclose(img, label[i])


class CustomAugmentation(HDF5Dataset):
    def get_transforms(self, phase):
        seed = 47
        raw_transform = Compose([
            transforms.RandomFlip(np.random.RandomState(seed)),
            transforms.RandomRotate90(np.random.RandomState(seed)),
            transforms.ToTensor(expand_dims=True)
        ])
        label_transform = Compose([
            transforms.RandomFlip(np.random.RandomState(seed)),
            transforms.RandomRotate90(np.random.RandomState(seed)),
            transforms.ToTensor(expand_dims=False)
        ])

        return raw_transform, label_transform


def create_random_dataset(shape):
    tmp_file = NamedTemporaryFile(delete=False)

    with h5py.File(tmp_file.name, 'w') as f:
        f.create_dataset('raw', data=np.random.rand(*shape))
        f.create_dataset('label', data=np.random.randint(0, 2, shape))

    return tmp_file.name
