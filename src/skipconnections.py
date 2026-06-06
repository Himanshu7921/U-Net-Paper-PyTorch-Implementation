import torch
import torch.nn as nn
from imagecropper import ImageCropper
class SkipConnection(nn.Module):
    """
    Implements the skip connection mechanism described in the U-Net architecture.

    U-Net uses skip connections to transfer high-resolution feature maps from
    the encoder to the decoder. Since valid convolutions reduce spatial dimensions,
    encoder feature maps are center-cropped before being concatenated with the
    corresponding decoder feature maps.

    Paper:
        U-Net: Convolutional Networks for Biomedical Image Segmentation

    Paper Link:
        https://arxiv.org/pdf/1505.04597

    Input:
        x: Encoder feature map of shape [B, C, H_x, W_x]
        y: Decoder feature map of shape [B, C, H_y, W_y]

    Output:
        Concatenated feature map of shape [B, 2C, H_y, W_y]

    Workflow:
        1. Crop the encoder feature map (x) to match the spatial dimensions of y.
        2. Concatenate the cropped encoder features and decoder features
           along the channel dimension.
    """
    def __init__(self, ImageCropper: ImageCropper):
        super(SkipConnection, self).__init__()
        self.img_cropper = ImageCropper()
    
    def forward(self, x: torch.tensor, y: torch.tensor):
        """
        Crops `x` to match the spatial dimensions of `y`, then concatenates the two tensors.
        Assumes that both tensors already have matching batch and channel dimensions

        Input: x: [B, C, H_x, W_x] and y: [B, C, H_y, W_y]
        Output: x: [B, 2C, H_y, W_y]
        """
        cropped_x = self.img_cropper(x, y)
        return torch.cat([cropped_x, y], dim = 1)