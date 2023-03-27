import torch

from pytorch3dunet.unet3d.buildingblocks import ResNetBlock
from pytorch3dunet.unet3d.model import UNet2D, UNet3D, ResidualUNet3D, ResidualUNetSE3D


class TestModel:
    def test_unet2d(self):
        model = UNet2D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)

    def test_unet3d(self):
        model = UNet3D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 32, 64, 64)
        with torch.no_grad():
            y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)

    def test_resnet_block1(self):
        blk = ResNetBlock(32, 64, is3d=False, order='cgr')
        blk.eval()
        x = torch.rand(1, 32, 64, 64)

        with torch.no_grad():
            y = blk(x)

        assert torch.all(0 <= y)

    def test_resnet_block2(self):
        blk = ResNetBlock(32, 32, is3d=False, order='cgr')
        blk.eval()
        x = torch.rand(1, 32, 64, 64)

        with torch.no_grad():
            y = blk(x)

        assert torch.all(0 <= y)

    def test_resunet3d(self):
        model = ResidualUNet3D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 32, 64, 64)
        y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)

    def test_resunetSE3d(self):
        model = ResidualUNetSE3D(1, 1, f_maps=16, final_sigmoid=True)
        model.eval()
        x = torch.rand(1, 1, 32, 64, 64)
        y = model(x)

        assert torch.all(0 <= y) and torch.all(y <= 1)
