import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch
from skimage.metrics import adapted_rand_error
from torch.utils.data import DataLoader

from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset
from pytorch3dunet.datasets.utils import default_prediction_collate, get_test_loaders
from pytorch3dunet.predict import _get_predictor
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.predictor import EmbeddingsPredictor
from pytorch3dunet.unet3d.utils import remove_halo


class FakePredictor(EmbeddingsPredictor):
    def __init__(self, model, output_dir, config, clustering, iou_threshold=0.7, **kwargs):
        super().__init__(model, output_dir, config, clustering, iou_threshold=iou_threshold, **kwargs)

    def _embeddings_to_segmentation(self, embeddings):
        return embeddings


class FakeModel:
    def __call__(self, input):
        return input

    def eval(self):
        pass


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

    def test_embeddings_predictor(self, tmpdir):
        config = {
            'model': {'output_heads': 1},
            'device': torch.device('cpu')
        }

        slice_builder_config = {
            'name': 'SliceBuilder',
            'patch_shape': (64, 200, 200),
            'stride_shape': (40, 150, 150)
        }

        transformer_config = {
            'raw': [
                {'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}
            ]
        }

        gt_file = 'resources/sample_ovule.h5'

        dataset = StandardHDF5Dataset(gt_file, phase='test',
                                      slice_builder_config=slice_builder_config,
                                      transformer_config=transformer_config,
                                      mirror_padding=None,
                                      raw_internal_path='label')

        loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=default_prediction_collate)

        predictor = FakePredictor(FakeModel(), tmpdir, config, clustering='meanshift', bandwidth=0.5)

        predictor(loader)

        with h5py.File(gt_file, 'r') as f:
            with h5py.File(os.path.join(tmpdir, 'sample_ovule_predictions.h5'), 'r') as g:
                gt = f['label'][...]
                segm = g['segmentation/meanshift'][...]
                arand_error = adapted_rand_error(gt, segm)[0]

                assert arand_error < 0.1

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
