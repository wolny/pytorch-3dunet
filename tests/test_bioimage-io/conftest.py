import os
from pathlib import Path

import pytest


@pytest.fixture
def cache_path(tmp_path):
    return Path(os.getenv("PYBIO_CACHE_PATH", tmp_path))
