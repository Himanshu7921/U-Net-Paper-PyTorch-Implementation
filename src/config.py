"""
Configuration file for my U-Net implementation based on the paper:

U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/pdf/1505.04597

Contains model architecture, dataset paths, image dimensions,
and training hyperparameters used throughout the project.
"""

from imagecropper import ImageCropper

config = {
    # ---------------------- Encoder settings ---------------------------
    "kernel_size_E": 3,
    "padding_E": 0,
    "max_pool_stride": 2,
    "max_pool_kernel_size": 2,
    "n_layers_E": 4,
    "hidden_channels_E": 64, # used in 1st Layer of Encoder

    # ---------------------- Bottleneck settings ---------------------------
    "kernel_size_B": 3,
    "padding_B": 0,
    "stride_B": 1,
    "hidden_channels_B": 1024,
    "skip_connections_channel_B": 512,

    # ---------------------- Decoder settings ------------------------------
    "kernel_size_D": 3,
    "padding_D": 0,
    "up_conv_stride": 2,
    "up_conv_kernel_size": 2,
    "n_layers_D": 3, # because 1 layer is now a bottleneck
    "hidden_channels_D": 512, # used in 1st Layer of Decoder

    # ---------------------- Image settings ----------------------
    "img_channel_dim": 1,
    "img_H": 316,
    "img_W": 316,

    # ---------------------- Dataset path ------------------------
    "image_dir": "./data/oxford-iiit-pet/images/images",
    "mask_dir": "./data/oxford-iiit-pet/annotations/annotations/trimaps",
    "output_dir": "./predictions",
    "model_save_dir": "./checkpoints",

    # Image Cropper
    "ImageCropper": ImageCropper,

    # Image Segmentation settings
    "n_classes": 2,
    "segmentation_map_hidden_channels": 64,
    "segmentation_map_kernel_size": 1,
    "mask_img_H": 132,
    "mask_img_W": 132,

    # U-Net Model specific settings
    "in_channels_E": 1, # encoder layer in_channels
    "in_channels_B": 512, # bottelneck layer in_channels
    "in_channels_D": 1024, # decoder layer in_channels
    "in_channels_S": 128, # output segmentation layer in_channels

    # ---------------------- Model Training settings----------------------
    "batch_size": 1,
    "lr": 1e-4,
    "optimizer": "adam",
    "epochs": 50,
    "device": "cuda",
}