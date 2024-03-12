import random

import torch

from pytorch3dunet.unet3d.config import load_config, copy_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import get_logger
import torch.distributed as dist
import torch.multiprocessing as mp

logger = get_logger('TrainingSetup')


def main(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method="tcp://127.0.0.1:23456")

    # Load and log experiment configuration
    config, config_path = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # Create trainer
    trainer = create_trainer(config, rank)
    # Copy config file
    copy_config(config, config_path)
    # Start training
    trainer.fit()

    dist.destroy_process_group()


if __name__ == '__main__':
    main(0, 1)
