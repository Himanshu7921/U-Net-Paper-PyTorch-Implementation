import torch
import torch.nn as nn
from typing import List
from config import config
from blocks import DecoderBlock
class Decoder(nn.Module):
       """
       Stacking Multiple DecoderBlocks to make decoder
       Each Decoder Block will reduce the channel_dim by half and progressevely increases the spatial resolution
       """
       def __init__(self, in_channels: int,
                kernel_size: int = config["kernel_size_D"],
                padding: int = config["padding_D"],
                n_layers: int = config["n_layers_D"],
                debug: bool = False,
                ):
              
              super(Decoder, self).__init__()
              self.in_channels = in_channels
              self.kernel_size = kernel_size
              self.padding = padding
              self.n_layers = n_layers
              self.debug = debug
              
              self.decoder_layers = []

              prev_channel = config["hidden_channels_D"]
              for _ in range(self.n_layers):
                     self.decoder_layers.append(
                            DecoderBlock(
                                   in_channels = prev_channel * 2, # because of concatenation
                                   hidden_channels_D = prev_channel // 2,
                                   out_channels = prev_channel // 2,
                                   debug = self.debug
                            )
                     )
                     prev_channel //= 2
              self.decoder = nn.Sequential(*self.decoder_layers)
       
       def forward(self, x: torch.Tensor, uncropped_tensor_list: List):
           """
           Crop and add skip connections
           """
           z = self.n_layers
           uncropped_tensor_list = uncropped_tensor_list[-z:][::-1] # get the last 3 tensors and get them in revere order
           for i in range(self.n_layers): # [0, 1, 2]
              x = self.decoder[i](x, uncropped_tensor_list[i])
           return x