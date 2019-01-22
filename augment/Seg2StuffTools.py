import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Segmentation2Affinities3D:
    def __init__(self, offset=[1], scale=[1, 1, 1]):
        """
        Generates z, x, y affinities from 3D segmentation
        """
        super(Segmentation2Affinities3D, self).__init__()
        self.offset = offset
        self.scale  = np.array(scale)
        self._build_kernel()

    def _build_kernel(self):
        # TODO make gpu version
        kernel = torch.zeros(3, 1, 3, 3, 3, dtype=torch.float32)
        # z azis
        kernel[0, 0, 1, 1, 1] = 1
        kernel[0, 0, 2, 1, 1] = -1
        # x axis
        kernel[1, 0, 1, 1, 1] = 1
        kernel[1, 0, 1, 2, 1] = -1
        # y axis
        kernel[2, 0, 1, 1, 1] = 1
        kernel[2, 0, 1, 1, 2] = -1

        self.kernel = kernel

    def __call__(self, segmentation):
        if len(segmentation.size()) == 3:
            segmentation = segmentation.view(-1, 1,
                                             segmentation.size(0),
                                             segmentation.size(1),
                                             segmentation.size(2))
        else:
            print("wrong batch size, input must be 3D")
            exit(1)

        labels = torch.empty(3 * len(self.offset),
                             segmentation.size(2),
                             segmentation.size(3),
                             segmentation.size(4))

        for i, offset in enumerate(self.offset):
            effective_offset = list((offset * self.scale).astype(np.int))

            segmentation = segmentation.float()
            pad_size = [int(effective_offset[1]), int(effective_offset[1]),
                        int(effective_offset[2]), int(effective_offset[2]),
                        int(effective_offset[0]), int(effective_offset[0])]

            segmentation_pad = nn.functional.pad(segmentation,
                                                 pad_size,
                                                 mode='replicate')

            mask = nn.functional.conv3d(segmentation_pad.float(),
                                        self.kernel,
                                        dilation=effective_offset,
                                        padding=effective_offset)

            labels[i * 3:i * 3 + 3] = ~(torch.abs(mask) < 1)
        return labels.float()


class Segmentation2Pmap3D:
    def __init__(self, offset=(1, ), scale=(1, 1, 1)):
        """
        Generates Binary probability maps from 3D Segmentations
        """
        super(Segmentation2Pmap3D, self).__init__()
        self.offset = offset
        self.scale = np.array(scale)
        self._build_kernel()

    def _build_kernel(self):
        # TODO gpu
        kernel = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        # all azis
        kernel[0, 0, 1, 1, 1] = 3
        kernel[0, 0, 2, 1, 1] = -1
        kernel[0, 0, 1, 2, 1] = -1
        kernel[0, 0, 1, 1, 2] = -1

        self.kernel = kernel

    def __call__(self, segmentation):
        if len(segmentation.size()) == 3:
            segmentation = segmentation.view(-1, 1,
                                             segmentation.size(0),
                                             segmentation.size(1),
                                             segmentation.size(2))
        else:
            print("wrong batch size, input must be 3D")
            exit(1)

        labels = torch.empty(len(self.offset),
                             segmentation.size(2),
                             segmentation.size(3),
                             segmentation.size(4))

        for i, offset in enumerate(self.offset):
            effective_offset = list((offset * self.scale).astype(np.int))

            segmentation = segmentation.float()
            pad_size = [int(effective_offset[1]), int(effective_offset[1]),
                        int(effective_offset[2]), int(effective_offset[2]),
                        int(effective_offset[0]), int(effective_offset[0])]

            segmentation_pad = nn.functional.pad(segmentation,
                                             pad_size,
                                             mode='replicate')

            mask = nn.functional.conv3d(segmentation_pad.float(),
                                        self.kernel,
                                        dilation=effective_offset)

            labels[i] = ~(torch.abs(mask[0]) < 1)

        return labels.float()
