import torch
import torch.nn as nn
from config import config

class OutputSegmentationMap(nn.Module):
    """
    Implements the final segmentation head of the U-Net architecture.

    Converts decoder feature maps into pixel-wise class predictions using
    a final 1x1 convolution, producing the output segmentation map.

    Paper:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/pdf/1505.04597

    Input:
        x : Decoder feature map [B, C, H, W]

    Output:
        Segmentation logits [B, n_classes, H, W]
    """
    def __init__(self,
                in_channels: int,
                out_channels: int = config["n_classes"],
                hidden_channels: int = config["segmentation_map_hidden_channels"],
                kernel_size: int = config["segmentation_map_kernel_size"]):
        super(OutputSegmentationMap, self).__init__()
        conv_1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = config["kernel_size_E"],
            padding = config["padding_E"],
        )
        conv_2 = nn.Conv2d(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            kernel_size = config["kernel_size_E"],
            padding = config["padding_E"],
        )
        
        # 1x1 Conv: Instead of using Fully Connected Linear Layers, we are using 1 x 1 convolution layer
        conv_3 = nn.Conv2d(
            in_channels = hidden_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = config["padding_E"],
        )

        self.layers = nn.Sequential(
            conv_1, conv_2, conv_3
        )
        
    def forward(self, x: torch.Tensor):
        return self.layers(x)