import os
from pathlib import Path
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
        predictor = get_predictor(model, test_config)
        # run the model prediction on the entire dataset and save to the 'output_file' H5
        predictor(test_loader)
    return tmp


def _get_result_shape(result_path: str, dataset_name: str = 'predictions'):
    with h5py.File(result_path, 'r') as f:
        return f[dataset_name].shape


class TestPredictor:
    def test_3d_predictor(self, tmpdir, test_config):
        tmp = _run_prediction(test_config, tmpdir, shape=(32, 64, 64))
        output_filename = os.path.split(tmp.name)[1] + '_predictions.h5'
        output_path = Path(tmpdir) / output_filename
        assert output_path.exists()
        assert _get_result_shape(output_path) == (2, 32, 64, 64)

    def test_2d_predictor(self, tmpdir, test_config_2d):
        tmp = _run_prediction(test_config_2d, tmpdir, shape=(3, 1, 256, 256))
        output_filename = os.path.split(tmp.name)[1] + '_predictions.h5'
        output_path = Path(tmpdir) / output_filename
        assert output_path.exists()
        assert _get_result_shape(output_path) == (2, 1, 256, 256)

    def test_predictor_save_segmentation(self, tmpdir, test_config):
        test_config['predictor']['save_segmentation'] = True
        tmp = _run_prediction(test_config, tmpdir, shape=(32, 64, 64))
        output_filename = os.path.split(tmp.name)[1] + '_predictions.h5'
        output_path = Path(tmpdir) / output_filename
        assert output_path.exists()
        # check that the output segmentation is saved, with the channel dimension reduced via argmax operation
        assert _get_result_shape(output_path) == (32, 64, 64)
