import os
import random

# Fix for OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from pytorch3dunet.unet3d.config import copy_config, load_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger("TrainingSetup")


def main():
    """Main entry point for training 3D U-Net models.

    Loads configuration from command line arguments, sets random seeds if specified,
    creates a trainer instance, and starts the training process.
    """
    # Load and log experiment configuration
    config, config_path = load_config()
    logger.info(config)

    manual_seed = config.get("manual_seed", None)
    if manual_seed is not None:
        logger.info(f"Seed the RNG for all devices with {manual_seed}")
        logger.warning("Using CuDNN deterministic setting. This may slow down the training!")
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # Create trainer
    trainer = create_trainer(config)
    # Copy config file
    copy_config(config, config_path)
    # Start training
    trainer.fit()


if __name__ == "__main__":
    main()
