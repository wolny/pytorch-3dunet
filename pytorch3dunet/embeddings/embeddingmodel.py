import torch.nn as nn

from pytorch3dunet.unet3d.buildingblocks import create_encoders, create_decoders, DoubleConv, InterpolateUpsampling
from pytorch3dunet.unet3d.utils import number_of_features_per_level


class SigmoidActivation(nn.Module):
    def __init__(self):
        super(SigmoidActivation, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert isinstance(x, tuple), "Expected a tuple: outputs from the 2 heads of the network"
        seg_logits, embeddings = x
        # apply the sigmoid only to the logits and pass `embeddings` through
        return self.sigmoid(seg_logits), embeddings


class EmbeddingUNet(nn.Module):
    """
    Shares the encoder path between two decoder paths:
        1. Semantic segmentation decoder (used to produce the segmentation mask)
        2. Embedding decoder (used to produce pixel embeddings)
    """

    def __init__(self, in_channels, seg_out_channels, emb_out_channels, f_maps=32, layer_order='gcr',
                 num_groups=8, num_levels=4, testing=False,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, **kwargs):
        super(EmbeddingUNet, self).__init__()

        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create common encoder path
        self.encoders = create_encoders(in_channels, f_maps, DoubleConv, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # segmentation's path; do not upsample in the 1st decoder
        self.seg_decoders = create_decoders(f_maps, DoubleConv, conv_kernel_size, conv_padding, layer_order, num_groups,
                                            upsample=False)

        # embeddings' path; do not upsample in the 1st decoder
        self.embedding_decoders = create_decoders(f_maps, DoubleConv, conv_kernel_size, conv_padding, layer_order,
                                                  num_groups, upsample=False)

        # single upsampling layer shared between segmentation and embedding paths in order to avoid computing the upsampling twice
        self.upsampling = InterpolateUpsampling()

        # reduce the number of number of channels from the seg decoder to the `seg_out_channels`
        self.seg_final_conv = nn.Conv3d(f_maps[0], seg_out_channels, 1)

        # reduce the number of number of channels from the seg decoder to the `emb_out_channels`
        self.emb_final_conv = nn.Conv3d(f_maps[0], emb_out_channels, 1)

        # final activation proxy
        self.final_activation = SigmoidActivation()

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # upsample x before passing to the segmentation and embedding paths
        seg_x = emb_x = self.upsampling(encoders_features[0], x)

        # segmentation's path
        for decoder, encoder_features in zip(self.seg_decoders, encoders_features):
            seg_x = decoder(encoder_features, seg_x)
        seg_x = self.seg_final_conv(seg_x)

        # embeddings' path
        for decoder, encoder_features in zip(self.embedding_decoders, encoders_features):
            emb_x = decoder(encoder_features, emb_x)
        emb_x = self.emb_final_conv(emb_x)

        output = (seg_x, emb_x)

        # apply final_activation nly during prediction
        if self.testing:
            output = self.final_activation(output)

        return output
