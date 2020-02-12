from pathlib import Path

import numpy
import pytest
import torch

from pybio.spec import load_spec_and_kwargs, utils
from pytorch3dunet.unet3d.model import UNet3D


@pytest.fixture
def dummy_input():
    return [numpy.random.uniform(-2,2,[1, 1, 112, 202, 202]).astype(numpy.float32)]


def test_dummy_input(cache_path, dummy_input):
    spec_path = Path(__file__).parent / "../../bioimage-io/UNet3DArabidopsisOvules.model.yaml"
    assert spec_path.exists()

    pybio_model = load_spec_and_kwargs(str(spec_path), cache_path=cache_path)
    for dummy, spec in zip(dummy_input, pybio_model.spec.inputs):
        assert str(dummy.dtype) == spec.data_type
        assert dummy.shape == spec.shape

