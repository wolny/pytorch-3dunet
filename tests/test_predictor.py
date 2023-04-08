import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.predict import get_predictor
from pytorch3dunet.unet3d.model import get_model


def _run_prediction(test_config, tmpdir, shape):
    # Add output dir
    test_config['loaders']['output_dir'] = tmpdir
    # create random dataset
    tmp = NamedTemporaryFile(delete=False)
    with h5py.File(tmp.name, 'w') as f:
        f.create_dataset('raw', data=np.random.rand(*shape))
    # Add input file
    test_config['loaders']['test']['file_paths'] = [tmp.name]
    # Create the model with random weights
    model = get_model(test_config['model'])
    if torch.cuda.is_available():
        model.cuda()
    for test_loader in get_test_loaders(test_config):
        predictor = get_predictor(model, tmpdir, test_config)
        # run the model prediction on the entire dataset and save to the 'output_file' H5
        predictor(test_loader)
    return tmp


class TestPredictor:
    def test_3d_predictor(self, tmpdir, test_config):
        tmp = _run_prediction(test_config, tmpdir, shape=(32, 64, 64))

        assert os.path.exists(os.path.join(tmpdir, os.path.split(tmp.name)[1] + '_predictions.h5'))

    def test_2d_predictor(self, tmpdir, test_config_2d):
        tmp = _run_prediction(test_config_2d, tmpdir, shape=(3, 1, 256, 256))

        assert os.path.exists(os.path.join(tmpdir, os.path.split(tmp.name)[1] + '_predictions.h5'))
