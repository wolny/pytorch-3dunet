"""3D Squeeze and Excitation Modules.

3D Extensions of the following 2D squeeze and excitation blocks:
    1. Channel Squeeze and Excitation (https://arxiv.org/abs/1709.01507)
    2. Spatial Squeeze and Excitation (https://arxiv.org/abs/1803.02579)
    3. Channel and Spatial Squeeze and Excitation (https://arxiv.org/abs/1803.02579)

New Project & Excite block, designed specifically for 3D inputs.

Coded by Anne-Marie Rickmann (https://github.com/arickm)
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


class ChannelSELayer3D(nn.Module):
    """3D extension of Squeeze-and-Excitation (SE) block.

    Described in:
        - Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507
        - Zhu et al., AnatomyNet, arXiv:1808.05238

    Args:
        num_channels: Number of input channels.
        reduction_ratio: By how much should the num_channels should be reduced. Default: 2.
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(x)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """3D extension of SE block -- squeezing spatially and exciting channel-wise.

    Described in: Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully
    Convolutional Networks, MICCAI 2018.

    Args:
        num_channels: Number of input channels.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None):
        """Forward pass.

        Args:
            x: Input tensor with shape (batch_size, num_channels, D, H, W).
            weights: Weights for few shot learning.

        Returns:
            Output tensor.
        """
        # channel squeeze
        batch_size, channel, D, H, W = x.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(x, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """3D extension of concurrent spatial and channel squeeze & excitation.

    Described in: Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully
    Convolutional Networks, arXiv:1803.02579.

    Args:
        num_channels: Number of input channels.
        reduction_ratio: By how much should the num_channels should be reduced. Default: 2.
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor
