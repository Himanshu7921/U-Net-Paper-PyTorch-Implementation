
import torch
import torch.nn as nn
from config import config
from blocks import EncoderBlock

class Encoder(nn.Module):
       """
       This represents the entire Encoder Architecture discussed in the paper
       I'll stack multiple encoder blocks to build the encoder pipeline

       Paper name: U-Net: Convolutional Networks for Biomedical Image Segmentation
       Paper Link: https://arxiv.org/pdf/1505.04597

       So my Encoder Block will contain:
       1. 3x3 Conv
       2. 3x3 Conv
       3. Max pool Layer
       4. uncropped_intermediate_tensor before max pooling (used in skip-connection pipeline of U-Nets)
       """
       def __init__(self, in_channels: int,
                kernel_size: int = config["kernel_size_E"],
                padding: int = config["padding_E"],
                n_layers: int = config["n_layers_E"],
                debug: bool = False):
              super(Encoder, self).__init__()
              self.in_channels = in_channels
              self.kernel_size = kernel_size
              self.padding = padding
              self.n_layers = n_layers
              self.debug = debug
       
              self.encoder_layers = [ # This is responsible to transform the image dim from 3 to 64
                     EncoderBlock(
                                   in_channels = self.in_channels,
                                   hidden_channels = config["hidden_channels_E"],
                                   out_channels = config["hidden_channels_E"],
                                   kernel_size = self.kernel_size,
                                   padding = self.padding,
                                   debug = self.debug
                     )
              ]

              # Build the Encoder
              prev_channel = config["hidden_channels_E"] # 3
              for _ in range(self.n_layers - 1):
                     self.encoder_layers.append(
                            EncoderBlock(
                                   in_channels = prev_channel,
                                   hidden_channels = prev_channel * 2,
                                   out_channels = prev_channel * 2,
                                   kernel_size = self.kernel_size,
                                   padding = self.padding,
                                   debug = self.debug
                            )
                     )
                     prev_channel *= 2
              self.encoder = nn.Sequential(*self.encoder_layers)
                            
       def forward(self, x: torch.Tensor):
              """
              Save the encoder feature maps before max pooling reduces their spatial resolution.
              These high-resolution features will later be used in the skip connections.
              Before concatenation, they must be cropped to match the dimensions of the corresponding decoder feature maps.
              """
              self.uncropped_tensor_list = []
              for layer in self.encoder:
                     x, uncropped_tensor = layer(x)
                     self.uncropped_tensor_list.append(uncropped_tensor) # each tensor will have a different shape
              return x