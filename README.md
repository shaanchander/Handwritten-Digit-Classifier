# Handwritten Digit Classifier (Client-Side)

This project runs digit inference fully in the browser using ONNX Runtime Web.

## Run

No Python backend is required for inference.

1. Open `static/index.html` directly in your browser.
2. Draw a digit and click Predict.

## Model Files

- `static/models/model-cnn.pt`
- `static/models/model-fcnn.pt`
- `static/models/model-cnn.onnx`
- `static/models/model-fcnn.onnx`

## Training

```bash
python train_cnn.py
python train_fcnn.py
```

Both scripts save `.pt` files to `static/models/`.

## Export To ONNX

```bash
python export_onnx.py
```

This reads `.pt` files from `static/models/` and writes `.onnx` files to `static/models/`.