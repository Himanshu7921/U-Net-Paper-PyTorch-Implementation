import torch
import torch.nn as nn
from config import config
from encoder import Encoder
from decoder import Decoder
from bottleneck import Bottleneck
from imagecropper import ImageCropper
from skipconnections import SkipConnection
from segmentation import OutputSegmentationMap

class UNet(nn.Module):
    """
    Implementation of the U-Net architecture for biomedical image segmentation.

    This model is composed of five major components:

        1. Encoder
        2. Bottleneck
        3. Decoder
        4. Skip Connections
        5. Output Segmentation Map

    The encoder progressively extracts hierarchical features while reducing
    the spatial dimensions of the input image. Intermediate feature maps are
    stored and later reused by the decoder through skip connections.

    The bottleneck acts as a bridge between the encoder and decoder. In this
    implementation, the bottleneck performs feature extraction followed by an
    upsampling operation before passing the resulting feature map to the decoder.

    The decoder reconstructs spatial information through a series of convolution
    and upsampling operations. At each stage, skip connections concatenate
    cropped encoder feature maps with decoder feature maps, helping preserve
    fine-grained localization information that would otherwise be lost during
    downsampling.

    The final OutputSegmentationMap module applies two convolution layers
    followed by a 1×1 convolution to produce pixel-wise class predictions for
    semantic segmentation.

    Architecture:
        Input Image
            ↓
        Encoder
            ↓
        Bottleneck
            ↓
        Skip Connections
            ↓
        Decoder
            ↓
    Output Segmentation Map

    Reference:
        Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
        Link : https://arxiv.org/pdf/1505.04597

    Author:
        Himanshu Singh
        Research Engineer | 2026

    Implementation based on:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
    """
    def __init__(self,
        # ---------------------- Encoder Settings ----------------------
        in_channels_E: int,
        in_channels_B: int,
        in_channels_D: int,
        in_channels_S: int,
        n_layers_E: int = config["n_layers_E"],
        kernel_size_E: int = config["kernel_size_E"],
        padding_E: int = config["padding_E"],

        # ---------------------- Bottleneck layer Settings ----------------------
        hidden_channels_B: int = config["hidden_channels_B"],
        out_channels_B: int = config["skip_connections_channel_B"],
        kernel_size_B: int = config["kernel_size_B"],
        padding_B: int = config["padding_B"],
        stride_B: int = config["stride_B"],

        # ---------------------- Decoder Settings ----------------------
        kernel_size_D: int = config["kernel_size_D"],
        padding_D: int = config["padding_D"],
        n_layers_D: int = config["n_layers_D"],
        
        # ---------------------- SkipConnection Settings ----------------------
        img_cropper: ImageCropper = config["ImageCropper"],

        # ---------------------- OutputSegmentationMap Settings ----------------------
        out_channels_S: int = config["n_classes"],
        hidden_channels_S: int = config["segmentation_map_hidden_channels"],
        kernel_size_S: int = config["segmentation_map_kernel_size"]):

        super(UNet, self).__init__()

        # Create encoder
        self.encoder = Encoder(
            in_channels = in_channels_E,
            n_layers = n_layers_E,
            kernel_size = kernel_size_E,
            padding = padding_E
        )

        # Create Bottleneck layer
        self.bottleneck_layer = Bottleneck(
            in_channels = in_channels_B,
            hidden_channels = hidden_channels_B,
            out_channels = out_channels_B,
            kernel_size = kernel_size_B,
            padding = padding_B,
            stride = stride_B
        )

        # Create Decoder
        self.decoder = Decoder(
            in_channels = in_channels_D,
            kernel_size = kernel_size_D,
            padding = padding_D,
            n_layers = n_layers_D
        )

        # Create Skip-connections
        self.skip_connection = SkipConnection(
            ImageCropper = img_cropper
        )

        # Create OutputSegmentationMap layer
        self.segmentation_map = OutputSegmentationMap(
            in_channels = in_channels_S,
            hidden_channels = hidden_channels_S,
            out_channels = out_channels_S,
            kernel_size = kernel_size_S
        )
        
    def forward(self, x: torch.Tensor):
        enc_images = self.encoder(x)
        bottleneck_output = self.bottleneck_layer(enc_images)
        decoder_output = self.decoder(bottleneck_output, self.encoder.uncropped_tensor_list)
        
        # Apply skip connections to the last tensor of the Decoder output
        return self.segmentation_map(self.skip_connection(self.encoder.uncropped_tensor_list[0], decoder_output))