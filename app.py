from __future__ import annotations

import base64
import io
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from functions import DigitCNN, DigitClassifier, get_device, prepare_prediction_image


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Handwritten Digit Classifier")


def load_model(model_path: Path, model_type: str, device: torch.device) -> torch.nn.Module:
    if model_type == "fcnn":
        model = DigitClassifier().to(device)
    elif model_type == "cnn":
        model = DigitCNN().to(device)
    else:
        raise ValueError("model_type must be 'cnn' or 'fcnn'")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


device = get_device()
models = {
    "cnn": load_model(BASE_DIR / "model-cnn.pt", "cnn", device),
    "fcnn": load_model(BASE_DIR / "model-fcnn.pt", "fcnn", device),
}


def decode_canvas_image(image_data: str) -> Image.Image:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc

    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read image") from exc


def predict_image(image: Image.Image, model_type: str) -> dict[str, object]:
    model = models.get(model_type)
    if model is None:
        raise HTTPException(status_code=400, detail="model_type must be 'cnn' or 'fcnn'")

    tensor = prepare_prediction_image(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        prediction = int(torch.argmax(probs).item())

    return {
        "prediction": prediction,
        "confidence": float(probs[prediction].item()),
        "probabilities": [float(value) for value in probs.tolist()],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/predict/{model_type}")
def predict(model_type: str, payload: dict[str, str]) -> JSONResponse:
    image_data = payload.get("image")
    if not image_data:
        raise HTTPException(status_code=400, detail="Missing image field")

    image = decode_canvas_image(image_data)
    result = predict_image(image, model_type)
    return JSONResponse(result)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)