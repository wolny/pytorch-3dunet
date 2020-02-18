import os

import h5py
import numpy as np
import pytest
import yaml

TEST_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'resources',
)


@pytest.fixture
def ovule_label():
    path = os.path.join(TEST_FILES, 'sample_ovule.h5')
    with h5py.File(path, 'r') as f:
        return f['label'][...]


@pytest.fixture
def transformer_config():
    config_path = os.path.join(TEST_FILES, 'transformer_config.yml')
    return yaml.load(open(config_path, 'r'))


@pytest.fixture
def train_config():
    config_path = os.path.join(TEST_FILES, 'config_train.yml')
    return yaml.load(open(config_path, 'r'))


@pytest.fixture
def train_config_2d():
    config_path = os.path.join(TEST_FILES, 'config_train_2d.yml')
    return yaml.load(open(config_path, 'r'))


@pytest.fixture
def random_input(tmpdir):
    shape = (32, 128, 128)
    return _create_random_input(tmpdir, shape, min_label=0)


@pytest.fixture
def random_input_with_ignore(tmpdir):
    shape = (32, 128, 128)
    return _create_random_input(tmpdir, shape, min_label=-1)


def _create_random_input(tmpdir, shape, min_label):
    path = os.path.join(tmpdir, 'test.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('raw', data=np.random.rand(*shape))
        f.create_dataset('label', data=np.random.randint(min_label, 2, shape))
    return path
