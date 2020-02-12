from pathlib import Path

import numpy
import pytest
import torch

from pybio.spec import load_spec_and_kwargs, utils
from pytorch3dunet.unet3d.model import UNet3D


@pytest.fixture
def dummy_input():
    return [numpy.random.uniform(-2, 2, [1, 1, 112, 202, 202]).astype(numpy.float32)]


def test_dummy_input(cache_path, dummy_input):
    spec_path = Path(__file__).parent / "../../bioimage-io/UNet3DArabidopsisOvules.model.yaml"
    assert spec_path.exists()

    pybio_model = load_spec_and_kwargs(str(spec_path), cache_path=cache_path)
    for dummy, spec in zip(dummy_input, pybio_model.spec.inputs):
        assert str(dummy.dtype) == spec.data_type
        assert dummy.shape == spec.shape


def test_Net3DArabidopsisOvules_forward(cache_path, dummy_input):
    spec_path = Path(__file__).parent / "../../bioimage-io/UNet3DArabidopsisOvules.model.yaml"
    assert spec_path.exists()

    pybio_model = load_spec_and_kwargs(str(spec_path), cache_path=cache_path)
    assert pybio_model.spec.outputs[0].shape.reference_input == "raw"
    assert pybio_model.spec.outputs[0].shape.scale == (1, 1, 1, 1, 1)
    assert pybio_model.spec.outputs[0].shape.offset == (0, 0, 0, 0, 0)

    instance = utils.get_instance(pybio_model)
    assert isinstance(instance, UNet3D)
    out = instance(*[torch.from_numpy(di) for di in dummy_input])
    assert out.shape == pybio_model.spec.inputs[0].shape
    assert str(out.dtype).split(".")[-1] == pybio_model.spec.outputs[0].data_type
