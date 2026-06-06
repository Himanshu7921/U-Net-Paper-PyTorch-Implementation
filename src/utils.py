import wandb
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
from pathlib import Path
from config import config
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_total_params(encoder, bottleneck, decoder, output_segmentation_map):
    encoder_params = sum(p.numel() for p in encoder.parameters())
    bottleneck_params = sum(p.numel() for p in bottleneck.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    output_segmentation_map_params = sum(
        p.numel() for p in output_segmentation_map.parameters()
    )

    total_params = (
        encoder_params
        + bottleneck_params
        + decoder_params
        + output_segmentation_map_params
    )

    print("=" * 60)
    print("Model Parameter Statistics")
    print("=" * 60)
    print(f"{'Encoder':30s}: {encoder_params:>15,}")
    print(f"{'Bottleneck':30s}: {bottleneck_params:>15,}")
    print(f"{'Decoder':30s}: {decoder_params:>15,}")
    print(f"{'Segmentation Head':30s}: {output_segmentation_map_params:>15,}")
    print("-" * 60)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f} M)")
    print("=" * 60)

# ------------------------------------ Standard Metrics ----------------------------------- 
def iou(pred, target, eps=1e-6):
    pred = pred.bool()
    target = target.bool()

    intersection = (pred & target).sum()
    union = (pred | target).sum()

    return (intersection + eps) / (union + eps)

def dice(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()

    return (2 * intersection + eps) / (
        pred.sum() + target.sum() + eps
    )

def pixel_accuracy(pred, target):
    pred = pred.argmax(dim=1)
    correct = (pred == target).sum()
    total = target.numel()
    return correct.float() / total

# -------------------------------------- Dataset Loader --------------------------------------
class OxfordPetSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_paths = sorted(
            Path(image_dir).glob("*.jpg")
        )

        self.mask_paths = [
            Path(mask_dir) / f"{image_path.stem}.png"
            for image_path in self.image_paths
        ]

        self.image_transform = transforms.Compose([
            transforms.Resize((config["img_H"], config["img_W"])),
            transforms.ToTensor()
        ])

        # because the mask shapes are variable and i want to match the size of my U-net model's output
        self.mask_transform = transforms.Resize(
            (config["mask_img_H"], config["mask_img_W"]),
            interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(
            self.image_paths[idx]
        ).convert("L")

        mask = Image.open(
            self.mask_paths[idx]
        )

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Pet vs Background: converting it to a binary classification problem
        mask = torch.from_numpy(
            (np.array(mask) == 1).astype(np.int64)
        )

        return image, mask

# -------------------------------------- Visualization Script -------------------------------------- 
@torch.no_grad()
def visualize_predictions(
    model,
    dataset,
    device,
    n_samples=3
):
    model.eval()

    random_indices = random.sample(
        range(len(dataset)),
        k=min(n_samples, len(dataset))
    )

    fig, axes = plt.subplots(
        len(random_indices),
        3,
        figsize=(12, 4 * len(random_indices))
    )

    # Handle n_samples = 1
    if len(random_indices) == 1:
        axes = [axes]

    for row, idx in enumerate(random_indices):
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        pred = model(image_batch)
        pred = pred.argmax(dim=1).squeeze(0).cpu()
        axes[row][0].imshow(
            image.squeeze(0),
            cmap="gray"
        )
        axes[row][0].set_title(f"Input Image (idx={idx})")
        axes[row][0].axis("off")
        axes[row][1].imshow(
            mask,
            cmap="gray"
        )
        axes[row][1].set_title("Ground Truth")
        axes[row][1].axis("off")
        axes[row][2].imshow(
            pred,
            cmap="gray"
        )
        axes[row][2].set_title("Prediction")
        axes[row][2].axis("off")
    plt.tight_layout()
    model.train()
    return fig

def visualize_dataset(dataset):
    indices = random.sample(range(len(dataset)), 10)

    fig, axes = plt.subplots(10, 2, figsize=(8, 40))

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]

        # Convert image from [C,H,W] -> [H,W,C]
        image = image.permute(1, 2, 0)

        axes[row, 0].imshow(image.squeeze(0), cmap="gray")
        axes[row, 0].set_title(f"Image {idx}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title(f"Mask {idx}")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()

# --------------------------------------- Experiment Tracking --------------------------------------- 
# weights and bias for experiment tracking
def initialize_wandb():
    wandb.init(
        project="unet-oxford-pets",
        name="unet-from-scratch",
        config={
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "optimizer": config["optimizer"],
            "architecture": "U-Net"
        }
    )

# ---------------------------------------- Proper Weight initialization ---------------------------------------- 
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu"
        )
        if module.bias is not None:
            nn.init.constant_(
                module.bias,
                0
            )
    elif isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu"
        )
        if module.bias is not None:
            nn.init.constant_(
                module.bias,
                0
            )

# ------------------------------------------ Saving and Loading Scripts ------------------------------------------
def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    save_path
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        },
        save_path
    )

def load_checkpoint(model, optimizer, saved_model_path, device):
    def optimizer_to(optimizer, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    checkpoint = torch.load(
        saved_model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    optimizer_to(
        optimizer,
        device
    )

    start_epoch = checkpoint["epoch"]

    loss = checkpoint["loss"]
    return model, optimizer, start_epoch, loss

# For Inference
def load_model(path, model, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model