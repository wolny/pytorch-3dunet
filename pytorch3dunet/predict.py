import importlib
import os

from pytorch3dunet.unet3d.predictor import AbstractPredictor

# Fix for OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import TorchDevice, load_config
from pytorch3dunet.unet3d.model import get_model

logger = utils.get_logger("UNet3DPredict")


def get_predictor(model: nn.Module, config: dict) -> AbstractPredictor:
    """Create and return a predictor instance based on the configuration.

    Args:
        model: The trained model to use for prediction.
        config: Configuration dictionary containing predictor settings.

    Returns:
        A predictor instance (StandardPredictor or LazyPredictor).
    """
    output_dir = config["loaders"].get("output_dir", None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    predictor_config = config.get("predictor", {})
    class_name = predictor_config.get("name", "StandardPredictor")

    m = importlib.import_module("pytorch3dunet.unet3d.predictor")
    predictor_class = getattr(m, class_name)
    out_channels = config["model"].get("out_channels")
    return predictor_class(model, output_dir, out_channels, **predictor_config, device=config["device"])


def main():
    """Main entry point for prediction with 3D U-Net models.

    Loads configuration from command line arguments, creates the model, loads trained weights,
    runs predictions on test datasets, and computes evaluation metrics if specified.
    """
    # Load configuration
    config, _ = load_config()

    # Create the model
    model = get_model(config["model"])
    device = config.get("device", None)
    assert device, "Device not specified in the config file and could not be inferred automatically"
    logger.info(f"Using device: {device}")

    # Load model state
    model_path = config["model_path"]
    logger.info(f"Loading model from {model_path}...")
    utils.load_checkpoint(model_path, model)

    # use DataParallel if more than 1 GPU available
    if device == TorchDevice.CUDA and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs for prediction")
    model = model.to(device)

    logger.info("Creating predictor...")
    # create predictor instance
    predictor = get_predictor(model, config)

    metrics = []
    for test_loader in get_test_loaders(config):
        # run the model prediction on the test_loader and save the results in the output_dir
        metric = predictor(test_loader)
        if metric is not None:
            metrics.append(metric)

    if metrics:
        # average across loaders
        metrics = torch.Tensor(metrics)
        per_class_metrics = metrics.mean(dim=0)
        avg_metric = metrics.mean()
        logger.info(f"Per-class average metric: {per_class_metrics}")
        logger.info(f"Average metric: {avg_metric}")


if __name__ == "__main__":
    main()
