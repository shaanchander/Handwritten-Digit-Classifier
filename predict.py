import sys
from PIL import Image

import torch

from functions import DigitClassifier, get_device, get_predict_transform


MODEL_PATH = "model.pt"


def load_image(path: str) -> torch.Tensor:
    transform = get_predict_transform()

    image = Image.open(path)
    image = transform(image)

    # If input image is black digit on white, invert it (MNIST is white digit on black background)
    if image.mean() > 0.5:
        image = 1.0 - image

    return image.unsqueeze(0)  # add batch dimension


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image.png>")
        return

    image_path = sys.argv[1]

    device = get_device()

    model = DigitClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

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