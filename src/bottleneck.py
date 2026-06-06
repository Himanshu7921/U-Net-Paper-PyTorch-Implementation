import torch
import torch.nn as nn
from config import config

class Bottleneck(nn.Module):
    """
    Implements the bottleneck block of the U-Net architecture, which acts as a bridge between the Encoder and Decoder.

    Given the encoder output:

    enc_img = encoder(img)
    enc_img.shape = [1, 512, 32, 32]

    the bottleneck performs two convolutional operations followed by an up-convolution:

    [1, 512, 32, 32]
            ↓
    [1, 1024, 30, 30]
            ↓
    [1, 1024, 28, 28]
            ↓
    [1, 512, 64, 64]

    Unlike the original U-Net architecture, the up-convolution is included within the bottleneck block.
    The resulting feature map is then passed to the Decoder for the remaining upsampling and skip-connection operations.

    Reference:
    Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
    Link: https://arxiv.org/pdf/1505.04597
    """
    def __init__(self,
                in_channels: int,
                hidden_channels: int = config["hidden_channels_B"],
                out_channels: int = config["skip_connections_channel_B"],
                kernel_size: int = config["kernel_size_B"],
                padding: int = config["padding_B"],
                stride: int = config["stride_B"]):
        super(Bottleneck, self).__init__()
        conv_1 = nn.Conv2d(
            in_channels = in_channels, # 512
            out_channels = hidden_channels, # 1024
            kernel_size = kernel_size, # 3 x 3
            padding = padding, # 0
            stride = stride # 1
        )
        conv_2 = nn.Conv2d(
            in_channels = hidden_channels, # 1024
            out_channels = hidden_channels, # 1024
            kernel_size = kernel_size, # 3 x 3
            padding = padding, # 0
            stride = stride # 1
        )
        up_conv = nn.ConvTranspose2d(
                    in_channels = hidden_channels,
                    out_channels = out_channels,
                    kernel_size = config["up_conv_kernel_size"],
                    padding = config["padding_D"],
                    stride = config["up_conv_stride"],
        )
        self.bottleneck_block = nn.Sequential(
            conv_1, conv_2, up_conv
        )
    
    def forward(self, x: torch.Tensor):
        return self.bottleneck_block(x)