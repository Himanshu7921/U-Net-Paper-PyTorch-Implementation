import torch
import torch.nn as nn
class ImageCropper(nn.Module):
    """
    Implements the center-cropping operation used in the original U-Net paper.

    Since valid convolutions reduce spatial dimensions, encoder feature maps
    must be cropped before being concatenated with decoder feature maps through
    skip connections.

    Paper:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/pdf/1505.04597

    Input:
        cropping_img : Feature map to crop [B, C, H_c, W_c]
        target_img   : Reference feature map [B, C, H_t, W_t]

    Output:
        Center-cropped feature map with shape [B, C, H_t, W_t]
    """
    def __init__(self):
        super(ImageCropper, self).__init__()
    
    def forward(self, cropping_img: torch.Tensor, target_img: torch.Tensor):
        # Gather details for cropping the image
        _, _, H_c, W_c = cropping_img.shape # W_c: Width of cropping image, H_c: Height of cropping image
        _, _, H_t, W_t = target_img.shape # W_t: Width of target image, H_t: Height of target image

        # How much to crop
        H, W = (H_c - H_t) // 2, (W_c - W_t) // 2

        # Crop the img
        cropped_img = cropping_img[:, :, H: H_c - H, W: W_c - W]
        return cropped_img