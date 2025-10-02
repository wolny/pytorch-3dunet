import torch

from pytorch3dunet.unet3d.buildingblocks import ResNetBlock
from pytorch3dunet.unet3d.model import ResidualUNet2D, ResidualUNet3D, ResidualUNetSE3D, UNet2D, UNet3D


class TestModel:
    def test_unet2d(self):
        model = UNet2D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 65, 65)
        with torch.no_grad():
            y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)

    def test_unet3d(self):
        model = UNet3D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 33, 65, 65)
        with torch.no_grad():
            y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)

    def test_resnet_block1(self):
        blk = ResNetBlock(33, 64, is3d=False, order="cgr")
        blk.eval()
        x = torch.rand(1, 33, 65, 65)

        with torch.no_grad():
            y = blk(x)

        assert torch.all(0 <= y)

    def test_resnet_block2(self):
        blk = ResNetBlock(33, 32, is3d=False, order="cgr")
        blk.eval()
        x = torch.rand(1, 33, 65, 65)

        with torch.no_grad():
            y = blk(x)

        assert torch.all(0 <= y)

    def test_resunet3d(self):
        model = ResidualUNet3D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 33, 65, 65)
        y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)

    def test_resunet2d(self):
        model = ResidualUNet2D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 65, 65)
        with torch.no_grad():
            y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)

    def test_resunetSE3d(self):
        model = ResidualUNetSE3D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 33, 65, 65)
        y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)
