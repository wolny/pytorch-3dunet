import h5py
import numpy as np
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from torch.utils.data import DataLoader

from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset, traverse_h5_paths, LazyHDF5Dataset
from pytorch3dunet.datasets.utils import remove_padding, default_prediction_collate


class TestHDF5Dataset:
    def test_hdf5_dataset(self, transformer_config):
        path = create_random_dataset((128, 128, 128))

        patch_shapes = [(127, 127, 127), (69, 70, 70), (32, 64, 64)]
        stride_shapes = [(127, 127, 127), (69, 69, 69), (32, 64, 64)]

        phase = 'test'

        for patch_shape, stride_shape in zip(patch_shapes, stride_shapes):
            with h5py.File(path, 'r') as f:
                raw = f['raw'][...]
                label = f['label'][...]

                dataset = StandardHDF5Dataset(path, phase=phase,
                                              slice_builder_config=_slice_builder_conf(patch_shape, stride_shape),
                                              transformer_config=transformer_config[phase]['transformer'],
                                              raw_internal_path='raw',
                                              label_internal_path='label')

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

    def test_lazy_hdf5_dataset(self, transformer_config):
        path = create_random_dataset((128, 128, 128))

        patch_shapes = [(127, 127, 127), (69, 70, 70), (32, 64, 64)]
        stride_shapes = [(127, 127, 127), (69, 69, 69), (32, 64, 64)]
        halo_shape = (16, 32, 32)
        phase = 'test'

        for patch_shape, stride_shape in zip(patch_shapes, stride_shapes):
            with h5py.File(path, 'r') as f:
                raw_shape = f['raw'].shape
                label_shape = f['label'].shape

            dataset = LazyHDF5Dataset(path, phase=phase,
                                      slice_builder_config=_slice_builder_conf(patch_shape, stride_shape,
                                                                               halo_shape),
                                      transformer_config=transformer_config[phase]['transformer'],
                                      raw_internal_path='raw',
                                      label_internal_path='label')

            # create zero-arrays of the same shape as the original dataset in order to verify if every element
            # was visited during the iteration
            visit_raw = np.zeros(raw_shape)
            visit_label = np.zeros(label_shape)

            for (_, idx) in dataset:
                visit_raw[idx] = 1
                visit_label[idx] = 1

            # verify that every element was visited at least once
            assert np.all(visit_raw)
            assert np.all(visit_label)

    def test_augmentation(self, transformer_config, tmpdir):
        raw = np.random.rand(32, 96, 96)
        # assign raw to label's channels for ease of comparison
        label = np.stack([raw for _ in range(3)])
        # create temporary h5 file
        tmp_file = tmpdir / "test.h5"
        with h5py.File(tmp_file, 'w') as f:
            f.create_dataset('raw', data=raw)
            f.create_dataset('label', data=label)

        # set phase='train' in order to execute the train transformers
        phase = 'train'
        dataset = StandardHDF5Dataset(tmp_file, phase=phase,
                                      slice_builder_config=_slice_builder_conf((16, 64, 64), (8, 32, 32)),
                                      transformer_config=transformer_config[phase]['transformer'])

        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        for (img, label) in data_loader:
            for i in range(label.shape[0]):
                assert np.allclose(img, label[i])

    def test_traverse_file_paths(self, tmpdir):
        test_tmp_dir = os.path.join(tmpdir, 'test')
        os.mkdir(test_tmp_dir)

        expected_files = [
            os.path.join(tmpdir, 'f1.h5'),
            os.path.join(test_tmp_dir, 'f2.h5'),
            os.path.join(test_tmp_dir, 'f3.hdf'),
            os.path.join(test_tmp_dir, 'f4.hdf5'),
            os.path.join(test_tmp_dir, 'f5.hd5')
        ]
        # create expected files
        for ef in expected_files:
            with h5py.File(ef, 'w') as f:
                f.create_dataset('raw', data=np.random.randn(4, 4, 4))

        # make sure that traverse_file_paths runs correctly
        file_paths = [os.path.join(tmpdir, 'f1.h5'), test_tmp_dir]
        actual_files = traverse_h5_paths(file_paths)

        assert expected_files == actual_files

    def test_halo(self, tmpdir: Path):
        halo_shape = (1, 2, 3)

        # create temporary h5 file
        raw = np.random.rand(32, 96, 96)
        tmp_file = tmpdir / "test.h5"

        with h5py.File(tmp_file, 'w') as f:
            f.create_dataset('raw', data=raw)

        # halo only implemented with test phase
        phase = 'test'
        dataset = StandardHDF5Dataset(tmp_file, phase=phase,
                                      slice_builder_config=_slice_builder_conf((16, 64, 64), (8, 32, 32), halo_shape),
                                      transformer_config=_transformer_test_conf())
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=default_prediction_collate)

        # verify all patches have the correct halo added and removed
        for (input_batch, indices_batch) in data_loader:  # input_batch has NCDHW shape, indices_batch has length N
            for input_, indices in zip(input_batch, indices_batch):  # input_ has CDHW shape, indices is for DHW
                input_ = remove_padding(input_, halo_shape)
                assert np.allclose(input_[0], raw[indices])

    def test_random_scale(self, transformer_config):
        path = create_random_dataset((200, 200, 172))

        patch_shapes = [(172, 172, 172)]
        stride_shapes = [(28, 28, 28)]

        phase = 'train'

        for patch_shape, stride_shape in zip(patch_shapes, stride_shapes):
            dataset = StandardHDF5Dataset(path, phase=phase,
                                          slice_builder_config=_slice_builder_conf(patch_shape, stride_shape),
                                          transformer_config=transformer_config[phase]['transformer'],
                                          raw_internal_path='raw',
                                          label_internal_path='label',
                                          random_scale=20)

            for raw, label in dataset:
                if raw.ndim == 3:
                    assert raw.shape == patch_shape
                    assert label.shape == patch_shape
                else:
                    assert raw.shape[1:] == patch_shape
                    assert label.shape[1:] == patch_shape


def create_random_dataset(shape, ignore_index=False, raw_datasets=None, label_datasets=None):
    if label_datasets is None:
        label_datasets = ['label']
    if raw_datasets is None:
        raw_datasets = ['raw']

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


def _slice_builder_conf(patch_shape, stride_shape, halo_shape=(0, 0, 0)):
    return {
        'name': 'SliceBuilder',
        'patch_shape': patch_shape,
        'stride_shape': stride_shape,
        'halo_shape': halo_shape,
    }


def _transformer_test_conf():
    return {
        'raw': [{'name': 'ToTensor', 'expand_dims': True}],
        # 'label': [{'name': 'ToTensor', 'expand_dims': True}],
    }
