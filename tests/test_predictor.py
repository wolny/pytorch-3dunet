import collections

import torch
from torch.utils.data import DataLoader

from datasets.hdf5 import HDF5Dataset
from unet3d.predictor import EmbeddingsPredictor


class FakePredictor(EmbeddingsPredictor):
    def __init__(self, model, loader, output_file, config, iou_threshold=0.8, **kwargs):
        super().__init__(model, loader, output_file, config, iou_threshold, **kwargs)

    def _embeddings_to_segmentation(self, embeddings):
        return embeddings


class FakeModel:
    def __call__(self, input):
        return input

    def eval(self):
        pass


class TestPredictor:
    def test_embeddings_predictor(self):
        config = {
            'model': {'output_heads': 1},
            'device': torch.device('cpu')
        }

        t_config = {
            'test': {
                'raw': [
                    {'name': 'Normalize'},
                    {'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}
                ]

            }
        }

        def my_collate(batch):
            if isinstance(batch[0], torch.Tensor):
                return torch.stack(batch, 0)
            elif isinstance(batch[0], slice):
                return batch[0]
            elif isinstance(batch[0], collections.Sequence):
                transposed = zip(*batch)
                return [my_collate(samples) for samples in transposed]

        dataset = HDF5Dataset('/home/adrian/workspace/pytorch-3dunet/resources/sample_cells.h5', (100, 200, 200),
                              (60, 150, 150), phase='test',
                              transformer_config=t_config, raw_internal_path='label')

        loader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=my_collate)

        predictor = FakePredictor(FakeModel(), loader,
                                  '/home/adrian/workspace/pytorch-3dunet/resources/output_segmentation.h5', config)

        predictor.predict()


if __name__ == '__main__':
    tp = TestPredictor()
    tp.test_embeddings_predictor()
