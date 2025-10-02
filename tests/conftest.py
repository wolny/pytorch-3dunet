import os

# Fix for OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import h5py
import numpy as np
import pytest
import yaml

from pytorch3dunet.unet3d.config import TorchDevice

TEST_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "resources",
)


def pytest_addoption(parser):
    parser.addoption("--device", type=TorchDevice, help="torch device to run on (cpu, cuda, mps)", default="cpu")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def ovule_label():
    path = os.path.join(TEST_FILES, "sample_ovule.h5")
    with h5py.File(path, "r") as f:
        return f["label"][...]


@pytest.fixture
def transformer_config(device):
    config_path = os.path.join(TEST_FILES, "transformer_config.yml")
    config = yaml.safe_load(open(config_path))
    config["device"] = device
    return config


@pytest.fixture
def train_config(device):
    config_path = os.path.join(TEST_FILES, "config_train.yml")
    config = yaml.safe_load(open(config_path))
    config["device"] = device
    return config


@pytest.fixture
def test_config(device):
    config_path = os.path.join(TEST_FILES, "config_test.yml")
    config = yaml.safe_load(open(config_path))
    config["device"] = device
    return config


@pytest.fixture
def test_config_2d(device):
    config_path = os.path.join(TEST_FILES, "config_test_2d.yml")
    config = yaml.safe_load(open(config_path))
    config["device"] = device
    return config


@pytest.fixture
def train_config_2d(device):
    config_path = os.path.join(TEST_FILES, "config_train_2d.yml")
    config = yaml.safe_load(open(config_path))
    config["device"] = device
    return config


@pytest.fixture
def random_input(tmpdir):
    shape = (32, 128, 128)
    return _create_random_input(tmpdir, shape, min_label=0)


@pytest.fixture
def random_input_with_ignore(tmpdir):
    shape = (32, 128, 128)
    return _create_random_input(tmpdir, shape, min_label=-1)


def _create_random_input(tmpdir, shape, min_label):
    path = os.path.join(tmpdir, "test.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=np.random.rand(*shape))
        f.create_dataset("label", data=np.random.randint(min_label, 2, shape))
    return path
