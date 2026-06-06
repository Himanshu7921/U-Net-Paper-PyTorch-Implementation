from pathlib import Path
from config import config
import matplotlib.pyplot as plt 
from utils import visualize_predictions, load_model, initialize_weights
from utils import OxfordPetSegmentationDataset
from model import UNet

def generate():
    model = UNet(
        in_channels_E = config["in_channels_E"],
        in_channels_B = config["in_channels_B"],
        in_channels_D = config["in_channels_D"],
        in_channels_S = config["in_channels_S"]
    )
    model.apply(initialize_weights)
    model = load_model(model = model, path = "./checkpoints/unet_saved_model.pth")

    dataset = OxfordPetSegmentationDataset(
        image_dir = config["image_dir"],
        mask_dir = config["mask_dir"]
    )
    fig = visualize_predictions(
                model=model,
                dataset=dataset,
                device= "cpu"
            )
    
    return fig

def main():
    fig = generate()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / "prediction.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()