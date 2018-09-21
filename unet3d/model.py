import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss
            or BCELoss respectively)
        interpolate (bool): if True use F.interpolate for upsampling otherwise
            use ConvTranspose3d
        final_sigmoid (bool): if True apply element-wise torch.sigmoid after the
            final 1x1x1 convolution; set to True if nn.BCELoss is to be used
            to train the model
        batch_norm (bool): if True use BatchNorm3d before ReLU
    """

    def __init__(self, in_channels, out_channels, interpolate=True,
                 final_sigmoid=True, batch_norm=True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, 64, is_max_pool=False, batch_norm=batch_norm),
            Encoder(64, 128, batch_norm=batch_norm),
            Encoder(128, 256, batch_norm=batch_norm),
            Encoder(256, 512, batch_norm=batch_norm)
        ])

        self.decoders = nn.ModuleList([
            Decoder(256 + 512, 256, interpolate, batch_norm=batch_norm),
            Decoder(128 + 256, 128, interpolate, batch_norm=batch_norm),
            Decoder(64 + 128, 64, interpolate, batch_norm=batch_norm)
        ])

        # in the last layer a 1×1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(64, out_channels, 1)

        if final_sigmoid:
            self.final_sigmoid = nn.Sigmoid()
        else:
            self.final_sigmoid = None

    def forward(self, x):
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
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if self.final_sigmoid is not None:
            x = self.final_sigmoid(x)

        return x


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (BatchNorm3d+ReLU+Conv3d)
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (BatchNorm3d+ReLU+Conv3d) instead of (Conv3d+BatchNorm3d+ReLU) suggested
    in the 3DUnet paper https://arxiv.org/pdf/1606.06650.pdf, cause:
    1. In the BN paper, the authors suggested to use BN immediately before the nonlinearity
    2. torch implementation of _DenseLayer uses BN+ReLU+conv as well
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        batch_norm (bool): if True use BatchNorm3d before ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 batch_norm=True):
        super(DoubleConv, self).__init__()
        if in_channels < out_channels:
            # encoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        if batch_norm:
            # conv1
            self.add_module('norm1', nn.BatchNorm3d(conv1_in_channels))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv3d(conv1_in_channels,
                                               conv1_out_channels,
                                               kernel_size,
                                               padding=1))
            # conv2
            self.add_module('norm2', nn.BatchNorm3d(conv2_in_channels))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv3d(conv2_in_channels,
                                               conv2_out_channels,
                                               kernel_size,
                                               padding=1))
        else:
            # conv1
            self.add_module('conv1', nn.Conv3d(conv1_in_channels,
                                               conv1_out_channels,
                                               kernel_size,
                                               padding=1))
            self.add_module('relu1', nn.ReLU(inplace=True))
            # conv2
            self.add_module('conv2', nn.Conv3d(conv2_in_channels,
                                               conv2_out_channels,
                                               kernel_size,
                                               padding=1))
            self.add_module('relu2', nn.ReLU(inplace=True))


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        is_max_pool (bool): if True use MaxPool3d before DoubleConv
        max_pool_kernel_size (tuple): the size of the window to take a max over
        batch_norm (bool): if True use BatchNorm3d before ReLU
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
                 is_max_pool=True,
                 max_pool_kernel_size=(2, 2, 2), batch_norm=True):
        super(Encoder, self).__init__()
        self.max_pool = nn.MaxPool3d(
            kernel_size=max_pool_kernel_size) if is_max_pool else None
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size=conv_kernel_size,
                                      batch_norm=batch_norm)

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        interpolate (bool): if True use nn.Upsample for upsampling, otherwise
            learn ConvTranspose3d if you have enough GPU memory and ain't
            afraid of overfitting
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d
        batch_norm (bool): if True use BatchNorm3d before ReLU
    """

    def __init__(self, in_channels, out_channels, interpolate, kernel_size=3,
                 scale_factor=(2, 2, 2), batch_norm=True):
        super(Decoder, self).__init__()
        if interpolate:
            self.upsample = None
        else:
            # make sure that the output size reverses the MaxPool3d
            # D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0]
            self.upsample = nn.ConvTranspose3d(2 * out_channels,
                                               2 * out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size=kernel_size,
                                      batch_norm=batch_norm)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
        else:
            x = self.upsample(x)
        # concatenate encoder_features (encoder path) with the upsampled input
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x
