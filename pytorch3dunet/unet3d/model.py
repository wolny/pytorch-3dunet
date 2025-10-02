from torch import nn

from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, create_decoders, create_encoders
from pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level


class AbstractUNet(nn.Module):
    """Base class for standard and residual UNet.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output segmentation masks. Note that the number of out_channels might correspond
            to either different semantic classes or to different binary segmentation mask. It's up to the user
            of the class to interpret the out_channels and use the proper loss criterion during training
            (i.e. CrossEntropyLoss for multi-class or BCEWithLogitsLoss for two-class respectively).
        final_sigmoid: If True apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply
            nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing.
        basic_module: Basic model for the encoder/decoder (DoubleConv, ResNetBlock, etc.).
        f_maps: Number of feature maps at each level of the encoder. If it's an integer, the number of feature
            maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4. Default: 64.
        layer_order: Determines the order of layers in `SingleConv` module. E.g. 'crg' stands for
            GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info. Default: 'gcr'.
        num_groups: Number of groups for the GroupNorm. Default: 8.
        num_levels: Number of levels in the encoder/decoder path (applied only if f_maps is an int). Default: 4.
        is_segmentation: If True and the model is in eval mode, Sigmoid/Softmax normalization is applied after
            the final convolution. If False (regression problem) the normalization layer is skipped. Default: True.
        conv_kernel_size: Size of the convolving kernel in the basic_module. Default: 3.
        pool_kernel_size: The size of the pooling window. Default: 2.
        conv_padding: Zero-padding added to all three sides of the input. Default: 1.
        conv_upscale: Number of the convolution to upscale in encoder if DoubleConv. Default: 2.
        upsample: Algorithm used for decoder upsampling. Options are 'nearest', 'linear', 'bilinear', 'trilinear',
            'area' (InterpolateUpsampling), 'deconv' (TransposeConvUpsampling), or None (no upsampling).
            Default: 'default' (chooses automatically).
        dropout_prob: Dropout probability. Default: 0.1.
        is3d: If True the model is 3D, otherwise 2D. Default: True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid,
        basic_module,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if "g" in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(
            in_channels,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            conv_upscale,
            dropout_prob,
            layer_order,
            num_groups,
            pool_kernel_size,
            is3d,
        )

        # create decoder path
        self.decoders = create_decoders(
            f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample, dropout_prob, is3d
        )

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

    def forward(self, x, return_logits=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W) for 3D or (N, C, H, W) for 2D,
                              where N is the batch size, C is the number of channels,
                              D is the depth, H is the height, and W is the width.
            return_logits (bool): If True, returns both the output and the logits.
                                  If False, returns only the output. Default is False.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
                          If return_logits is True, returns a tuple of (output, logits).
        """
        output, logits = self._forward_logits(x)
        if return_logits:
            return output, logits
        return output

    def _forward_logits(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features, strict=False):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if self.final_activation is not None:
            # compute final activation
            out = self.final_activation(x)
            # return both probabilities and logits
            return out, x

        return x, x


class UNet3D(AbstractUNet):
    """3D U-Net model from "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation".

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder.
    Reference: https://arxiv.org/pdf/1606.06650.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
        )


class ResidualUNet3D(AbstractUNet):
    """Residual 3D U-Net model implementation.

    Uses ResNetBlock as a basic building block, summation joining instead of concatenation joining,
    and transposed convolutions for upsampling (watch out for block artifacts). Since the model
    effectively becomes a residual net, in theory it allows for deeper UNet.

    Reference: https://arxiv.org/pdf/1706.00120.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
        )


class ResidualUNetSE3D(AbstractUNet):
    """Residual 3D U-Net model implementation with Squeeze and Excitation blocks.

    Uses ResNetBlockSE as a basic building block, summation joining instead of concatenation joining,
    and transposed convolutions for upsampling (watch out for block artifacts). Since the model
    effectively becomes a residual net, in theory it allows for deeper UNet.

    Reference: https://arxiv.org/pdf/1706.00120.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlockSE,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
        )


class UNet2D(AbstractUNet):
    """2D U-Net model from "U-Net: Convolutional Networks for Biomedical Image Segmentation".

    Reference: https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=False,
        )


class ResidualUNet2D(AbstractUNet):
    """Residual 2D U-Net model implementation.

    Reference: https://arxiv.org/pdf/1706.00120.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=False,
        )


def get_model(model_config):
    model_class = get_class(model_config["name"], modules=["pytorch3dunet.unet3d.model"])
    return model_class(**model_config)


def is_model_2d(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    return isinstance(model, UNet2D)
