import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.trainer import UNetTrainer
from pytorch3dunet.unet3d.utils import TensorboardFormatter, create_lr_scheduler, create_optimizer


class TestUNet3DTrainer:
    def test_ce_loss(self, tmpdir, capsys, train_config):
        with capsys.disabled():
            assert_train_save_load(tmpdir, train_config, "CrossEntropyLoss", "MeanIoU", "UNet3D")

    def test_wce_loss(self, tmpdir, capsys, train_config):
        with capsys.disabled():
            assert_train_save_load(tmpdir, train_config, "WeightedCrossEntropyLoss", "MeanIoU", "UNet3D")

    def test_bce_loss(self, tmpdir, capsys, train_config):
        with capsys.disabled():
            assert_train_save_load(tmpdir, train_config, "BCEWithLogitsLoss", "DiceCoefficient", "UNet3D")

    def test_dice_loss(self, tmpdir, capsys, train_config):
        with capsys.disabled():
            assert_train_save_load(tmpdir, train_config, "DiceLoss", "MeanIoU", "UNet3D")

    def test_residual_unet(self, tmpdir, capsys, train_config):
        with capsys.disabled():
            assert_train_save_load(tmpdir, train_config, "CrossEntropyLoss", "MeanIoU", "ResidualUNet3D")

    def test_2d_unet(self, tmpdir, capsys, train_config_2d):
        with capsys.disabled():
            assert_train_save_load(tmpdir, train_config_2d, "CrossEntropyLoss", "MeanIoU", "UNet2D",
                                   shape=(3, 1, 128, 128))


def assert_train_save_load(tmpdir, train_config, loss, val_metric, model, weight_map=False, shape=(3, 64, 64, 64)):
    max_num_epochs = train_config["trainer"]["max_num_epochs"]
    log_after_iters = train_config["trainer"]["log_after_iters"]
    validate_after_iters = train_config["trainer"]["validate_after_iters"]
    max_num_iterations = train_config["trainer"]["max_num_iterations"]

    trainer = _train_save_load(tmpdir, train_config, loss, val_metric, model, weight_map, shape)

    assert trainer.num_iterations == max_num_iterations
    assert trainer.max_num_epochs == max_num_epochs
    assert trainer.log_after_iters == log_after_iters
    assert trainer.validate_after_iters == validate_after_iters
    assert trainer.max_num_iterations == max_num_iterations


def _train_save_load(tmpdir, train_config, loss, val_metric, model, weight_map, shape):
    binary_loss = loss in ["BCEWithLogitsLoss", "DiceLoss", "BCEDiceLoss", "GeneralizedDiceLoss"]

    train_config["model"]["name"] = model
    train_config.update({
        # get device to train on
        "loss": {"name": loss, "weight": np.random.rand(2).astype(np.float32), "pos_weight": 3.},
        "eval_metric": {"name": val_metric}
    })
    train_config["model"]["final_sigmoid"] = binary_loss

    if weight_map:
        train_config["loaders"]["weight_internal_path"] = "weight_map"

    loss_criterion = get_loss_criterion(train_config)
    eval_criterion = get_evaluation_metric(train_config)
    model = get_model(train_config["model"])
    model = model.to(train_config["device"])

    if loss in ["BCEWithLogitsLoss"]:
        label_dtype = "float32"
        train_config["loaders"]["train"]["transformer"]["label"][0]["dtype"] = label_dtype
        train_config["loaders"]["val"]["transformer"]["label"][0]["dtype"] = label_dtype

    train = _create_random_dataset(shape, binary_loss)
    val = _create_random_dataset(shape, binary_loss)
    train_config["loaders"]["train"]["file_paths"] = [train]
    train_config["loaders"]["val"]["file_paths"] = [val]

    loaders = get_train_loaders(train_config)

    optimizer = create_optimizer(train_config["optimizer"], model)
    lr_scheduler = create_lr_scheduler(train_config.get("lr_scheduler", None), optimizer)

    formatter = TensorboardFormatter()
    trainer = UNetTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, tmpdir,
                          max_num_epochs=train_config["trainer"]["max_num_epochs"],
                          max_num_iterations=train_config["trainer"]["max_num_iterations"],
                          validate_after_iters=train_config["trainer"]["log_after_iters"],
                          log_after_iters=train_config["trainer"]["log_after_iters"], tensorboard_formatter=formatter,
                          device=train_config["device"])
    trainer.fit()
    # test loading the trainer from the checkpoint
    trainer = UNetTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, tmpdir,
                          max_num_epochs=train_config["trainer"]["max_num_epochs"],
                          max_num_iterations=train_config["trainer"]["max_num_iterations"],
                          validate_after_iters=train_config["trainer"]["log_after_iters"],
                          log_after_iters=train_config["trainer"]["log_after_iters"], tensorboard_formatter=formatter,
                          resume=os.path.join(tmpdir, "last_checkpoint.pytorch"),
                          device=train_config["device"])
    return trainer


def _create_random_dataset(shape, channel_per_class):
    tmp = NamedTemporaryFile(delete=False)

    with h5py.File(tmp.name, "w") as f:
        l_shape = w_shape = shape
        # make sure that label and weight tensors are 3D
        if len(shape) == 4:
            l_shape = shape[1:]
            w_shape = shape[1:]

        if channel_per_class:
            l_shape = (2,) + l_shape

        f.create_dataset("raw", data=np.random.rand(*shape))
        f.create_dataset("label", data=np.random.randint(0, 2, l_shape))
        f.create_dataset("weight_map", data=np.random.rand(*w_shape))

    return tmp.name
