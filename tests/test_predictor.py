import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.predict import get_predictor
from pytorch3dunet.unet3d.model import get_model


def _run_prediction(test_config, tmpdir, shape):
    # Add output dir
    test_config["loaders"]["output_dir"] = tmpdir
    # create random dataset
    tmp = NamedTemporaryFile(delete=False)
    with h5py.File(tmp.name, "w") as f:
        f.create_dataset("raw", data=np.random.rand(*shape))
        f.create_dataset("label", data=np.random.randint(0, 2, shape).astype("uint8"))
    # Add input file
    test_config["loaders"]["test"]["file_paths"] = [tmp.name]
    # Create the model with random weights
    model = get_model(test_config["model"])

    model = model.to(test_config["device"])
    results = []
    for test_loader in get_test_loaders(test_config):
        predictor = get_predictor(model, test_config)
        # run the model prediction on the entire dataset and save to the 'output_file' H5
        result = predictor(test_loader)
        if result is not None:
            results.append(result)
    if results:
        return tmp, results
    return tmp


def _get_result_shape(result_path: str, dataset_name: str = "predictions"):
    with h5py.File(result_path, "r") as f:
        return f[dataset_name].shape


class TestPredictor:
    def test_3d_predictor(self, tmpdir, test_config):
        tmp = _run_prediction(test_config, tmpdir, shape=(32, 64, 64))
        output_filename = os.path.split(tmp.name)[1] + "_predictions.h5"
        output_path = Path(tmpdir) / output_filename
        assert output_path.exists()
        assert _get_result_shape(output_path) == (2, 32, 64, 64)

    def test_2d_predictor(self, tmpdir, test_config_2d):
        tmp = _run_prediction(test_config_2d, tmpdir, shape=(3, 1, 256, 256))
        output_filename = os.path.split(tmp.name)[1] + "_predictions.h5"
        output_path = Path(tmpdir) / output_filename
        assert output_path.exists()
        assert _get_result_shape(output_path) == (2, 1, 256, 256)

    def test_predictor_save_segmentation(self, tmpdir, test_config):
        test_config["predictor"]["save_segmentation"] = True
        tmp = _run_prediction(test_config, tmpdir, shape=(32, 64, 64))
        output_filename = os.path.split(tmp.name)[1] + "_predictions.h5"
        output_path = Path(tmpdir) / output_filename
        assert output_path.exists()
        # check that the output segmentation is saved, with the channel dimension reduced via argmax operation
        assert _get_result_shape(output_path) == (32, 64, 64)

    def test_performance_metric(self, tmpdir, test_config):
        test_config["predictor"]["save_segmentation"] = True
        test_config["predictor"]["performance_metric"] = "mean_iou"
        test_config["predictor"]["gt_internal_path"] = "label"
        tmp, results = _run_prediction(test_config, tmpdir, shape=(32, 64, 64))

        output_filename = os.path.split(tmp.name)[1] + "_predictions.h5"
        output_path = Path(tmpdir) / output_filename
        assert output_path.exists()
        # check that the output segmentation is saved, with the channel dimension reduced via argmax operation
        assert _get_result_shape(output_path) == (32, 64, 64)
        # assert results array is non-zero
        assert np.mean(results) > 0

    def test_lazy_predictor(self, tmpdir, test_config):
        test_config["predictor"]["name"] = "LazyPredictor"
        tmp = _run_prediction(test_config, tmpdir, shape=(32, 64, 64))
        output_filename = os.path.split(tmp.name)[1] + "_predictions.h5"
        output_path = Path(tmpdir) / output_filename
        assert output_path.exists()
        # check that the output segmentation is saved, with the channel dimension reduced via argmax operation
        assert _get_result_shape(output_path) == (2, 32, 64, 64)
        # assert results array is non-zero
        with h5py.File(output_path, "r") as f:
            assert np.count_nonzero(f["predictions"][...]) > 0

    def test_performance_metric_lazy_predictor(self, tmpdir, test_config):
        test_config["predictor"]["name"] = "LazyPredictor"
        test_config["predictor"]["save_segmentation"] = True
        test_config["predictor"]["performance_metric"] = "mean_iou"
        test_config["predictor"]["gt_internal_path"] = "label"
        tmp, results = _run_prediction(test_config, tmpdir, shape=(32, 64, 64))

        output_filename = os.path.split(tmp.name)[1] + "_predictions.h5"
        output_path = Path(tmpdir) / output_filename
        assert output_path.exists()
        # check that the output segmentation is saved, with the channel dimension reduced via argmax operation
        assert _get_result_shape(output_path) == (32, 64, 64)
        # assert results array is non-zero
        assert np.mean(results) > 0
