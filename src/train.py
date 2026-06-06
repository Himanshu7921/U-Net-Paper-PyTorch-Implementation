import torch
import wandb
import torch.nn as nn
from model import UNet
from pathlib import Path
from config import config
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils import OxfordPetSegmentationDataset, pixel_accuracy, iou, dice, save_checkpoint, initialize_weights, initialize_wandb

def train_model(
    model: UNet,
    optimizer: torch.optim.Adam,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = config["epochs"],
    start_epoch = 0
):
    print("\n[Initializing weight and bias for Experiment Tracking.....]\n")
    initialize_wandb()

    print(f"Training the Model on device = [{device}]")
    model = model.to(device)
    wandb.watch(
        model,
        log="all",
        log_freq=100
    )

    loss_fn = nn.CrossEntropyLoss()

    epoch_bar = tqdm(
        range(start_epoch, epochs),
        desc="Training",
        unit="epoch",
        colour="green"
    )
    output_dir = Path(config["checkpoints"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in epoch_bar:

        model.train()

        # ---------------- Epoch Metrics ----------------
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
        # ------------------------------------------------

        batch_bar = tqdm(
            train_loader,
            leave=False,
            desc=f"Epoch {epoch + 1}/{epochs}",
            colour="blue"
        )

        for images, masks in batch_bar:

            images = images.to(device)
            masks = masks.to(device)

            # ------------------------------------------------
            # Forward Pass
            # ------------------------------------------------

            pred_masks = model(images)

            loss = loss_fn(
                pred_masks,
                masks.long()
            )


            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            pred_classes = pred_masks.argmax(dim=1)
            batch_loss = loss.item()
            batch_acc = pixel_accuracy(
                pred_masks,
                masks
            )
            batch_iou = iou(
                pred_classes,
                masks
            )
            batch_dice = dice(
                pred_classes,
                masks
            )

            # Accumulate Epoch Metrics
            epoch_loss += batch_loss
            epoch_acc += batch_acc.item()
            epoch_iou += batch_iou.item()
            epoch_dice += batch_dice.item()

            # Batch Progress Bar
            p_classes = pred_masks.argmax(dim=1).unique(
                return_counts=True
            )
            batch_bar.set_postfix(
                loss=f"{batch_loss:.4f}",
                dice=f"{batch_dice:.4f}",
                iou=f"{batch_iou:.4f}",
                pred_classes = f"{p_classes}"
            )

        # Epoch Metrics
        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)
        epoch_iou /= len(train_loader)
        epoch_dice /= len(train_loader)

        # W&B Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "pixel_accuracy": epoch_acc,
            "iou": epoch_iou,
            "dice": epoch_dice,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        # Epoch Progress Bar
        p_classes = pred_masks.argmax(dim=1).unique(
                return_counts=True
            )
        epoch_bar.set_postfix(
            loss=f"{epoch_loss:.4f}",
            acc=f"{epoch_acc:.4f}",
            dice=f"{epoch_dice:.4f}",
            iou=f"{epoch_iou:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            p_classes = f"{p_classes}"
        )

        # Visualization Every 1 Epochs
        if epoch % 1 == 0:
            # Save the checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=epoch_loss,
                save_path=f"./checkpoints/unet_epoch_{epoch+1}.pth"
            )
            print(
                f"batch_loss = {batch_loss:.4f}"
            )

        # Console Logging
        print(
            f"[Epoch {epoch+1:03d}/{epochs}] "
            f"Loss={epoch_loss:.4f} "
            f"Acc={epoch_acc:.4f} "
            f"Dice={epoch_dice:.4f} "
            f"IoU={epoch_iou:.4f}"
        )
        print(
            pred_masks.argmax(dim=1).unique(
                return_counts=True
            )
        )

    print(
        f"\nTraining Finished | Final Loss = {epoch_loss:.4f}"
    )
    wandb.finish()
    return model, epoch_loss

def main():
    # Get a device to train on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the Dataset
    dataset = OxfordPetSegmentationDataset(
        image_dir = config["image_dir"],
        mask_dir = config["mask_dir"]
    )

    # Define the Model and apply Weight Initializations
    model = UNet(
        in_channels_E = config["in_channels_E"],
        in_channels_B = config["in_channels_B"],
        in_channels_D = config["in_channels_D"],
        in_channels_S = config["in_channels_S"]
    )
    model.apply(initialize_weights)

    # Create a DataLoader for training with batch_size = 1
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    if config["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(params = model.parameters(), lr = config["lr"])
    else:
        # Silently uses SGD Optimizer
        optimizer = torch.optim.SGD(params = model.parameters(), lr = config["lr"])
    
    # Train the Model
    trained_model, loss = train_model(model = model, optimizer = optimizer, device = device, train_loader = train_loader)

    # Save the Model
    print(f"\nSaving the Model | Location: [checkpoints/unet_epoch_final.pth]")
    save_checkpoint(
                model = trained_model,
                optimizer=optimizer,
                epoch = config["epochs"],
                loss = loss,
                save_path=f"./checkpoints/unet_saved_model.pth"
    )

if __name__ == "__main__":
    main()