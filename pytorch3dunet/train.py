import torch

from pytorch3dunet.datasets.utils import get_class
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('TrainingSetup')


def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # create trainer
    default_trainer_builder_class = 'UNet3DTrainerBuilder'
    trainer_builder_class = config['trainer'].get('builder', default_trainer_builder_class)
    trainer_builder = get_class(trainer_builder_class, modules=['pytorch3dunet.unet3d.trainer'])
    trainer = trainer_builder.build(config)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
