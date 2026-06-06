import torch
import torch.nn as nn
from config import config
from skipconnections import SkipConnection

class EncoderBlock(nn.Module):
    """
    This represents one Block of the Encoder

    > In the original paper they have used a 3x3 convolution followed by another 3x3 convolution followed by a Max-polling layer
    Paper name: U-Net: Convolutional Networks for Biomedical Image Segmentation
    Paper Link: https://arxiv.org/pdf/1505.04597

    So my Encoder Block will contain:
        1. 3x3 Conv (Padding = 0, Stride = 1, kernel_size = 3)
        2. 3x3 Conv (Padding = 0, Stride = 1, kernel_size = 3)
        3. Max pool Layer (Stride = 2, kernel_size = 2)
        4. Stores the `uncropped_intermediate_tensor` before max pooling. During decoding,
        these tensors are cropped as needed and used in the skip connections to
        concatenate encoder and decoder feature maps
    """
    def __init__(self, in_channels: int,
            hidden_channels: int,
            out_channels: int,
            kernel_size: int = config["kernel_size_E"],
            padding: int = config["padding_E"],
            debug: bool = False):
            super(EncoderBlock, self).__init__()
            self.relu = nn.ReLU()
            self.conv_1 = nn.Conv2d(in_channels = in_channels,
                    out_channels = hidden_channels,
                    kernel_size = kernel_size,
                    padding = padding)
            self.conv_2 = nn.Conv2d(in_channels = hidden_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    padding = padding)
            self.max_pool = nn.MaxPool2d(
                    stride = config["max_pool_stride"], # 2
                    kernel_size = config["max_pool_kernel_size"] # 2
            )
            self.debug = debug
                
    def forward(self, x: torch.Tensor):
        """
        Note: I am intentionally storing all intermediate encoder feature maps before applying max pooling.
        These feature maps will later be used for skip connections in the decoder.

        I also store them in a list because the required crop size is not known beforehand.
        The amount of cropping depends on the spatial dimensions of the corresponding decoder feature maps,
        which become available only during the decoding stage.
        """
        if self.debug: # Debug Mode is on, print the intermediate tensor shapes
            x = self.relu(self.conv_1(x))
            print(f"> Inside Encoder Block")
            print(f"x.shape = {x.shape}")
            print("-" * 70)
            uncropped_intermediate_tensor = self.relu(self.conv_2(x))
            print(f"uncropped_intermediate_tensor.shape = {uncropped_intermediate_tensor.shape}")
            print("-" * 70)
            y = self.max_pool(uncropped_intermediate_tensor)
            print(f"after_max_pool.shape = {y.shape}")
            print("-" * 70)
            return y, uncropped_intermediate_tensor
        else:
            x = self.relu(self.conv_1(x))
            uncropped_intermediate_tensor = self.relu(self.conv_2(x))
            y = self.max_pool(uncropped_intermediate_tensor)
            return y, uncropped_intermediate_tensor
            
class DecoderBlock(nn.Module):
    """
    This represents one Block of the Decoder

    > In the original paper they have used a (3x3 convolution) followed by another (3x3 convolution) followed by a (2x2 Transposed Convolution)
    Paper name: U-Net: Convolutional Networks for Biomedical Image Segmentation
    Paper Link: https://arxiv.org/pdf/1505.04597

    So my Decoder Block will contain:
        1. 3x3 Conv (out_channel = in_channel // 2, padding = 0, kernel_size = 3)
            [Reduces both the channel dim and spatial resolution]
        2. 3x3 Conv (out_channel = in_channel // 2, padding = 0, kernel_size = 3)
            [Reduces both the channel dim and spatial resolution]
        3. 2x2 up-conv (Transposed Convolution) [out_channel = in_channel // 2, kernel_size = 2, stride = 2]
            {Increases the spatial resolution, but decreases the channel_dim by half}
        4. Crops the encoder feature maps saved before max pooling and concatenates them
        with the corresponding decoder feature maps. Each crop-and-concatenate
        operation is performed before the first 3×3 convolution in the decoder block.
    """
    def __init__(self, in_channels: int,
                out_channels: int,
                hidden_channels_D: int = config["hidden_channels_D"],
                kernel_size_D: int = config["kernel_size_D"],
                padding_D: int = config["padding_D"],
                debug: bool = False):
              super(DecoderBlock, self).__init__()

              self.ImageCropper = config["ImageCropper"]
              self.skip_connection = SkipConnection(self.ImageCropper)

              self.relu = nn.ReLU()
            #   print("in_channels:", in_channels, type(in_channels))
            #   print("out_channels:", hidden_channels_D, type(hidden_channels_D))
            #   print("kernel_size:", kernel_size_D, type(kernel_size_D))
            #   print("padding:", padding_D, type(padding_D))

              # 3x3 convolution
              self.conv_1 = nn.Conv2d(
                  in_channels = in_channels,
                  out_channels = hidden_channels_D,
                  kernel_size = kernel_size_D, # 3
                  padding = padding_D # 0
              )
                            # 3x3 convolution
              self.conv_2 = nn.Conv2d(
                  in_channels = hidden_channels_D,
                  out_channels = hidden_channels_D,
                  kernel_size = kernel_size_D, # 3
                  padding = padding_D # 0
              )
              self.up_conv = nn.ConvTranspose2d(
                  in_channels = hidden_channels_D,
                  out_channels = out_channels,
                  kernel_size = config["up_conv_kernel_size"], # 2
                  stride = config["up_conv_stride"] # 2
              )
              self.debug = debug

    def forward(self, x: torch.Tensor, uncropped_intermediate_tensor: torch.Tensor):
        if self.debug: # Debug Mode is on, print the intermediate tensor shapes
            print("-" * 60)
            print(f"x.shape = {x.shape}")
            print(f"uncropped_intermediate_tensor.shape = {uncropped_intermediate_tensor.shape}")
            x = self.skip_connection(uncropped_intermediate_tensor, x)
            print(f"x.shape after skip connections = {x.shape}")
            print("-" * 60)
            x = self.relu(self.conv_1(x))
            x = self.relu(self.conv_2(x))
            y = self.up_conv(x)
            return y
        else:
            x = self.skip_connection(uncropped_intermediate_tensor, x)
            x = self.relu(self.conv_1(x))
            x = self.relu(self.conv_2(x))
            y = self.up_conv(x)
            return y
              