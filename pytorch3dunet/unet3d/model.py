import torch.nn as nn

from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders
from pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level

class AbstractEncoder(nn.Module):
    def __init__(self, in_channels, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, is3d=True):
        super(AbstractEncoder, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d)

        self.num_features = f_maps[-1]

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        return x, encoders_features

class AbstractDecoder(nn.Module):
    def __init__(self, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, upsampling_conv_kernel_size=3, scale_factor=(2, 2, 2),
                 conv_padding=1, upsampling_padding=1, is3d=True):
        super(AbstractDecoder, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, upsampling_conv_kernel_size, scale_factor, conv_padding, upsampling_padding, layer_order, num_groups,
                                        is3d)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

        self.num_features = f_maps[-1]
        self.out_channels = out_channels

    def forward(self, x, encoders_features):

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x

class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, upsampling_conv_kernel_size=3, scale_factor=(2, 2, 2), pool_kernel_size=2,
                 conv_padding=1, upsampling_padding=1, is3d=True):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoder = AbstractEncoder(in_channels, basic_module, f_maps, layer_order,
                 num_groups, num_levels, conv_kernel_size, pool_kernel_size,
                 conv_padding, is3d)

        # create decoder path
        self.decoder = AbstractDecoder(out_channels, final_sigmoid, basic_module, f_maps, layer_order,
                 num_groups, num_levels, is_segmentation, conv_kernel_size, upsampling_conv_kernel_size, scale_factor,
                 conv_padding, upsampling_padding, is3d)

        self.num_features = f_maps[-1]
        self.out_channels = out_channels

    def forward(self, x):
        # encoder part
        x, encoders_features = self.encoder(x)

        feature_vec = encoders_features[0]

        # decoder part
        x = self.decoder(x, encoders_features)

        return x, feature_vec

class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=conv_kernel_size,
                                     conv_padding=conv_padding,
                                     is3d=True)

class Encoder3D(AbstractEncoder):
    def __init__(self, in_channels, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, pool_kernel_size=2, conv_kernel_size=3, conv_padding=1, **kwargs):
        super(Encoder3D, self).__init__(in_channels=in_channels,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     pool_kernel_size=pool_kernel_size,
                                     conv_kernel_size=conv_kernel_size,
                                     conv_padding=conv_padding,
                                     is3d=True)

class Decoder3D(AbstractDecoder):
    def __init__(self, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, conv_padding=1, **kwargs):
        super(Decoder3D, self).__init__(out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=conv_kernel_size,
                                     conv_padding=conv_padding,
                                     is3d=True)

class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_kernel_size=3, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_kernel_size=conv_kernel_size,
                                             conv_padding=conv_padding,
                                             is3d=True)

class ResidualEncoder3D(AbstractEncoder):
    def __init__(self, in_channels, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, pool_kernel_size=2, conv_kernel_size=3, conv_padding=1, **kwargs):
        super(ResidualEncoder3D, self).__init__(in_channels=in_channels,
                                     basic_module=ResNetBlock,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     pool_kernel_size=pool_kernel_size,
                                     conv_kernel_size=conv_kernel_size,
                                     conv_padding=conv_padding,
                                     is3d=True)

class ResidualDecoder3D(AbstractDecoder):
    def __init__(self, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, upsampling_conv_kernel_size=3, scale_factor=(2, 2, 2), conv_padding=1, upsampling_padding=1, **kwargs):
        super(ResidualDecoder3D, self).__init__(out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=ResNetBlock,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=conv_kernel_size,
                                     upsampling_conv_kernel_size=upsampling_conv_kernel_size,
                                     scale_factor=scale_factor,
                                     conv_padding=conv_padding,
                                     upsampling_padding=upsampling_padding,
                                     is3d=True)

class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_kernel_size=3, conv_padding=1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               final_sigmoid=final_sigmoid,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               is_segmentation=is_segmentation,
                                               conv_kernel_size=conv_kernel_size,
                                               conv_padding=conv_padding,
                                               is3d=True)

class ResidualEncoderSE3D(AbstractEncoder):
    def __init__(self, in_channels, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, pool_kernel_size=2, conv_kernel_size=3, conv_padding=1, **kwargs):
        super(ResidualEncoderSE3D, self).__init__(in_channels=in_channels,
                                     basic_module=ResNetBlockSE,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     pool_kernel_size=pool_kernel_size,
                                     conv_kernel_size=conv_kernel_size,
                                     conv_padding=conv_padding,
                                     is3d=True)

class ResidualDecoderSE3D(AbstractDecoder):
    def __init__(self, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, upsampling_conv_kernel_size=3, scale_factor=(2, 2, 2), conv_padding=1, upsampling_padding=1, **kwargs):
        super(ResidualDecoderSE3D, self).__init__(out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=ResNetBlockSE,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=conv_kernel_size,
                                     upsampling_conv_kernel_size=upsampling_conv_kernel_size,
                                     scale_factor=scale_factor,
                                     conv_padding=conv_padding,
                                     upsampling_padding=upsampling_padding,
                                     is3d=True)

class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, conv_padding=1, upsampling_padding=1, **kwargs):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=conv_kernel_size,
                                     conv_padding=conv_padding,
                                     upsampling_padding=upsampling_padding,
                                     is3d=False)

class Encoder2D(AbstractEncoder):
    def __init__(self, in_channels, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, conv_kernel_size=3, conv_padding=1, **kwargs):
        super(Encoder2D, self).__init__(in_channels=in_channels,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     conv_padding=conv_padding,
                                     conv_kernel_size=conv_kernel_size,
                                     is3d=False)

class Decoder2D(AbstractDecoder):
    def __init__(self, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, conv_padding=1, **kwargs):
        super(Decoder2D, self).__init__(out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=conv_kernel_size,
                                     conv_padding=conv_padding,
                                     is3d=False)

def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'pytorch3dunet.unet3d.model',
        'pytorch3dunet.unet3d.unetr',
    ])
    return model_class(**model_config)
