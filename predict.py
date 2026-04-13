import sys
from PIL import Image

import torch

from functions import (
    DigitClassifier,
    DigitCNN,
    get_device,
    prepare_prediction_image,
)


def load_image(path: str) -> torch.Tensor:
    image = Image.open(path)
    image = prepare_prediction_image(image)
    return image.unsqueeze(0)  # add batch dimension


def load_model(model_path: str, model_type: str, device: torch.device) -> torch.nn.Module:
    """Load the specified model type from the given path."""
    if model_type == "fcnn":
        model = DigitClassifier().to(device)
    elif model_type == "cnn":
        model = DigitCNN().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'fcnn' or 'cnn'.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def main():
    if len(sys.argv) != 4:
        print("Usage: python predict.py <model_path> <model_type> <image.png>")
        print("  model_type: 'fcnn' or 'cnn'")
        return

    model_path = sys.argv[1]
    model_type = sys.argv[2]
    image_path = sys.argv[3]

    device = get_device()
    model = load_model(model_path, model_type, device)
    image = load_image(image_path).to(device)

    with torch.no_grad():
        logits = model(image)
        prediction = logits.argmax(dim=1).item()
        probs = torch.softmax(logits, dim=1)
        print(probs)
        print()

    print(f"Predicted digit: {prediction}")


if __name__ == "__main__":
    main()