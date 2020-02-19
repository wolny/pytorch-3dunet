import os

import h5py
import numpy as np
import torch
from skimage.metrics import adapted_rand_error
from torch.utils.data import DataLoader

from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset
from pytorch3dunet.datasets.utils import prediction_collate
from pytorch3dunet.unet3d.predictor import EmbeddingsPredictor
from pytorch3dunet.unet3d.utils import remove_halo


class FakePredictor(EmbeddingsPredictor):
    def __init__(self, model, loader, output_file, config, clustering, iou_threshold=0.7, **kwargs):
        super().__init__(model, loader, output_file, config, clustering, iou_threshold=iou_threshold, **kwargs)

    def _embeddings_to_segmentation(self, embeddings):
        return embeddings


class FakeModel:
    def __call__(self, input):
        return input

    def eval(self):
        pass


class TestPredictor:
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
        output_file = os.path.join(tmpdir, 'output_segmentation.h5')

        dataset = StandardHDF5Dataset(gt_file, phase='test',
                                      slice_builder_config=slice_builder_config,
                                      transformer_config=transformer_config,
                                      mirror_padding=None,
                                      raw_internal_path='label')

        loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=prediction_collate)

        predictor = FakePredictor(FakeModel(), loader, output_file, config, clustering='meanshift', bandwidth=0.5)

        predictor.predict()

        with h5py.File(gt_file, 'r') as f:
            with h5py.File(output_file, 'r') as g:
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
