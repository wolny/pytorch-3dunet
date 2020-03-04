from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
from zipfile import ZipFile

import pytest

from pybio.spec.utils import cache_uri, get_instance, load_model

MODEL_EXTENSIONS = (".model.yaml", ".model.yml")
PACKAGE_URL = "https://github.com/wolny/pytorch-3dunet/releases/download/1.2.6/UNet3DArabidopsisOvules.model.zip"


def guess_model_path(file_names: List[str]) -> Optional[str]:
    for file_name in file_names:
        if file_name.endswith(MODEL_EXTENSIONS):
            return file_name

    return None


def eval_model_zip(model_zip: ZipFile, cache_path: Path):
    with TemporaryDirectory() as tempdir:
        temp_path = Path(tempdir)
        if cache_path is None:
            cache_path = temp_path / "cache"

        model_zip.extractall(temp_path)
        spec_file_str = guess_model_path([str(file_name) for file_name in temp_path.glob("*")])
        pybio_model = load_model(spec_file_str, root_path=temp_path, cache_path=cache_path)

        return get_instance(pybio_model)


@pytest.fixture
def package_bytes(cache_path):
    return cache_uri(uri_str=PACKAGE_URL, hash={}, cache_path=cache_path)


def test_eval_model_zip(package_bytes, cache_path):
    with ZipFile(package_bytes) as zf:
        model = eval_model_zip(zf, cache_path=cache_path)

    assert model  # todo: improve test_eval_model_zip
