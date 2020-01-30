import os

import h5py
import torch
from torch.utils.data import DataLoader

from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset, prediction_collate
from pytorch3dunet.unet3d.predictor import EmbeddingsPredictor
from pytorch3dunet.unet3d.utils import adapted_rand


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
                                      raw_internal_path='label')

        loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=prediction_collate)

        predictor = FakePredictor(FakeModel(), loader, output_file, config, clustering='meanshift', bandwidth=0.5)

        predictor.predict()

        with h5py.File(gt_file, 'r') as f:
            with h5py.File(output_file, 'r') as g:
                gt = f['label'][...]
                segm = g['segmentation/meanshift'][...]
                arand_error = adapted_rand(segm, gt)

                assert arand_error < 0.1
