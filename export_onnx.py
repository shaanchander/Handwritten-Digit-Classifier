from __future__ import annotations
import base64
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


def export_embedded_model_payload(model_name: str, source_path: Path, target_path: Path) -> None:
    encoded = base64.b64encode(source_path.read_bytes()).decode("ascii")
    content = (
        "window.EMBEDDED_MODELS = window.EMBEDDED_MODELS || {};\n"
        f"window.EMBEDDED_MODELS.{model_name} = \"{encoded}\";\n"
    )
    target_path.write_text(content, encoding="utf-8")
    print(f"Exported embedded payload {model_name}: {target_path}")


def main() -> None:
    output_dir = BASE_DIR / "static" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    cnn_onnx_path = output_dir / "model-cnn.onnx"
    fcnn_onnx_path = output_dir / "model-fcnn.onnx"

    export_model(
        "cnn",
        DigitCNN(),
        output_dir / "model-cnn.pt",
        cnn_onnx_path,
    )
    export_model(
        "fcnn",
        DigitClassifier(),
        output_dir / "model-fcnn.pt",
        fcnn_onnx_path,
    )

    export_embedded_model_payload("cnn", cnn_onnx_path, output_dir / "model-cnn.onnx.b64.js")
    export_embedded_model_payload("fcnn", fcnn_onnx_path, output_dir / "model-fcnn.onnx.b64.js")


if __name__ == "__main__":
    main()