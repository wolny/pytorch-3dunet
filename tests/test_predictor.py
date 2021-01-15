import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.predict import _get_predictor
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.utils import remove_halo


class TestPredictor:
    def test_stanard_predictor(self, tmpdir, test_config):
        # Add output dir
        test_config['loaders']['output_dir'] = tmpdir

        # create random dataset
        tmp = NamedTemporaryFile(delete=False)

        with h5py.File(tmp.name, 'w') as f:
            shape = (32, 64, 64)
            f.create_dataset('raw', data=np.random.rand(*shape))

        # Add input file
        test_config['loaders']['test']['file_paths'] = [tmp.name]

        # Create the model with random weights
        model = get_model(test_config['model'])
        # Create device and update config
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        test_config['device'] = device
        model = model.to(device)

        for test_loader in get_test_loaders(test_config):
            predictor = _get_predictor(model, tmpdir, test_config)
            # run the model prediction on the entire dataset and save to the 'output_file' H5
            predictor(test_loader)

        assert os.path.exists(os.path.join(tmpdir, os.path.split(tmp.name)[1] + '_predictions.h5'))

    def test_remove_halo(self):
        patch_halo = (4, 4, 4)
        shape = (128, 128, 128)
        input = np.random.randint(0, 10, size=(1, 16, 16, 16))

        index = (slice(0, 1), slice(12, 28), slice(16, 32), slice(16, 32))
        u_patch, u_index = remove_halo(input, index, shape, patch_halo)

        assert np.array_equal(input[:, 4:12, 4:12, 4:12], u_patch)
        assert u_index == (slice(0, 1), slice(16, 24), slice(20, 28), slice(20, 28))

        index = (slice(0, 1), slice(112, 128), slice(112, 128), slice(112, 128))
        u_patch, u_index = remove_halo(input, index, shape, patch_halo)

        assert np.array_equal(input[:, 4:16, 4:16, 4:16], u_patch)
        assert u_index == (slice(0, 1), slice(116, 128), slice(116, 128), slice(116, 128))
