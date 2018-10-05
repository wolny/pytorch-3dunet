import torch
import torch.nn as nn


# Based on: https://github.com/kuangliu/pytorch-groupnorm

class _GroupNorm(nn.Module):
    dim_to_params_shape = {
        3: (1, 1, 1, 1, 1),
        2: (1, 1, 1, 1),
        1: (1, 1, 1)
    }

    def __init__(self, num_features, dim, num_groups=32, eps=1e-5):
        super(_GroupNorm, self).__init__()
        assert dim in [1, 2, 3], f'Unsupported dimensionality: {dim}'
        params_shape = list(self.dim_to_params_shape[dim])
        params_shape[1] = num_features
        self.weight = nn.Parameter(torch.ones(params_shape))
        self.bias = nn.Parameter(torch.zeros(params_shape))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        self._check_input_dim(x)
        # save original shape
        shape = x.size()

        N = shape[0]
        C = shape[1]
        G = self.num_groups
        assert C % G == 0, 'Channel dim must be multiply of number of groups'

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()

        # restore original shape
        x = x.view(shape)
        return x * self.weight + self.bias

    def _check_input_dim(self, x):
        raise NotImplementedError


class GroupNorm3d(_GroupNorm):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm3d, self).__init__(num_features, 3, num_groups, eps)

    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError(f'Expected 5D input (got {x.dim()}D input)')


class GroupNorm2d(_GroupNorm):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm2d, self).__init__(num_features, 2, num_groups, eps)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f'Expected 4D input (got {x.dim()}D input)')


class GroupNorm1d(_GroupNorm):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm1d, self).__init__(num_features, 1, num_groups, eps)

    def _check_input_dim(self, x):
        if x.dim() != 3:
            raise ValueError(f'Expected 3D input (got {x.dim()}D input)')
