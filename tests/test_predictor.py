import os

import h5py
import torch
from torch.utils.data import DataLoader

from datasets.hdf5 import HDF5Dataset, prediction_collate
from unet3d.predictor import EmbeddingsPredictor
from unet3d.utils import adapted_rand


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

        t_config = {
            'test': {
                'raw': [
                    {'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}
                ]

            }
        }

        gt_file = 'resources/sample_cells.h5'
        output_file = os.path.join(tmpdir, 'output_segmentation.h5')
        dataset = HDF5Dataset(gt_file, (100, 200, 200), (60, 150, 150), phase='test',
                              transformer_config=t_config, raw_internal_path='label')

        loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=prediction_collate)

        predictor = FakePredictor(FakeModel(), loader, output_file, config, clustering='meanshift', bandwidth=0.5)

        predictor.predict()

        with h5py.File(gt_file, 'r') as f:
            with h5py.File(output_file, 'r') as g:
                gt = f['label'][...]
                segm = g['segmentation/meanshift'][...]
                arand_error = adapted_rand(segm, gt)

                assert arand_error < 0.1
