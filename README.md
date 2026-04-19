# Handwritten Digit Classifier

Train MNIST digit classifiers in PyTorch, export to ONNX, and run inference fully in the browser

## Background

This repo compares two model types for handwritten digit recognition:

- CNN model (`DigitCNN`) for spatial pattern learning
- FCNN model (`DigitClassifier`) as a baseline

Models are trained in Python, exported to ONNX, and loaded in a static frontend for side-by-side predictions.

## Project Structure

- `train_cnn.py`: trains CNN and saves `static/models/model-cnn.pt`
- `train_fcnn.py`: trains FCNN and saves `static/models/model-fcnn.pt`
- `export_onnx.py`: exports `.pt` checkpoints to `.onnx` and generates embedded base64 model payloads
- `functions.py`: shared model definitions, transforms, and preprocessing utilities
- `static/`: browser app (`index.html`, `app.js`, `style.css`) and model artifacts

## Requirements

- Python 3.14+
- `uv` (recommended) or `pip`

## Setup

From the project root:

```bash
uv sync
```

## Train Models

```bash
python train_cnn.py
python train_fcnn.py
```

This will download MNIST (if needed) and write checkpoints to:

- `static/models/model-cnn.pt`
- `static/models/model-fcnn.pt`

## Export to ONNX

```bash
python export_onnx.py
```

This generates:

- `static/models/model-cnn.onnx`
- `static/models/model-fcnn.onnx`
- `static/models/model-cnn.onnx.b64.js`
- `static/models/model-fcnn.onnx.b64.js`

## Run the Web App

Open `static/index.html` in your browser.

Then:

1. Draw a single digit on the canvas.
2. Click `Predict`.
3. Compare CNN vs FCNN predictions and confidence.

Inference runs client-side in your browser (no Python backend needed at runtime).