from __future__ import annotations
from pathlib import Path
import torch
from functions import DigitCNN, DigitClassifier


BASE_DIR = Path(__file__).resolve().parent


def export_model(model_name: str, model: torch.nn.Module, source_path: Path, target_path: Path) -> None:
    state_dict = torch.load(source_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        target_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        external_data=False,
    )
    print(f"Exported {model_name}: {target_path}")


def main() -> None:
    output_dir = BASE_DIR / "static" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    export_model(
        "cnn",
        DigitCNN(),
        BASE_DIR / "model-cnn.pt",
        output_dir / "model-cnn.onnx",
    )
    export_model(
        "fcnn",
        DigitClassifier(),
        BASE_DIR / "model-fcnn.pt",
        output_dir / "model-fcnn.onnx",
    )


if __name__ == "__main__":
    main()