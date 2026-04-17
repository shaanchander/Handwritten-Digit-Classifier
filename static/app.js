const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const resultsGridEl = document.getElementById("resultsGrid");
const modelStatusEl = document.getElementById("modelStatus");
const themeToggleBtn = document.getElementById("theme-toggle") || document.getElementById("themeToggle");
const darkSchemeQuery = window.matchMedia("(prefers-color-scheme: dark)");

const MODEL_PATHS = {
  cnn: "models/model-cnn.onnx",
  fcnn: "models/model-fcnn.onnx",
};

const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;
const MNIST_IMAGE_SIZE = 28;
const MNIST_DIGIT_TARGET_SIZE = 22;
const FOREGROUND_THRESHOLD = 40;
const NORMALIZED_BLACK = (0 - MNIST_MEAN) / MNIST_STD;

let drawing = false;
let currentTheme = "light";
let userOverrodeTheme = false;
let modelsLoaded = false;
let predictionInFlight = false;
let modelSessions = {
  cnn: null,
  fcnn: null,
};

function decodeBase64ToUint8Array(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

async function createInferenceSession(modelType) {
  const options = { executionProviders: ["wasm"] };
  const embeddedModel = window.EMBEDDED_MODELS?.[modelType];

  if (embeddedModel) {
    const modelBytes = decodeBase64ToUint8Array(embeddedModel);
    return window.ort.InferenceSession.create(modelBytes, options);
  }

  return window.ort.InferenceSession.create(MODEL_PATHS[modelType], options);
}

function setModelStatus(message, state = "loading") {
  if (!modelStatusEl) {
    return;
  }

  modelStatusEl.textContent = message;
  modelStatusEl.dataset.state = state;
}

function setPredictEnabled(enabled) {
  predictBtn.disabled = !enabled;
}

function setTheme(theme) {
  currentTheme = theme;
  document.documentElement.setAttribute("data-theme", theme);
  if (themeToggleBtn) {
    const isDark = theme === "dark";
    themeToggleBtn.setAttribute("aria-pressed", String(isDark));
    themeToggleBtn.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
    themeToggleBtn.setAttribute("title", isDark ? "Switch to light mode" : "Switch to dark mode");
  }
}

function getCanvasPalette() {
  if (currentTheme === "dark") {
    return {
      background: "#000000",
      stroke: "#ffffff",
      blankThreshold: 5,
      blankComparator: "gt",
    };
  }

  return {
    background: "#ffffff",
    stroke: "#111111",
    blankThreshold: 250,
    blankComparator: "lt",
  };
}

function resizeCanvas() {
  const palette = getCanvasPalette();
  context.fillStyle = palette.background;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = palette.stroke;
  context.lineWidth = 24;
}

function updateCanvasDrawingStyles() {
  const palette = getCanvasPalette();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = palette.stroke;
  context.lineWidth = 24;
}

function invertCanvasPixels() {
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = imageData.data;

  for (let index = 0; index < pixels.length; index += 4) {
    pixels[index] = 255 - pixels[index];
    pixels[index + 1] = 255 - pixels[index + 1];
    pixels[index + 2] = 255 - pixels[index + 2];
  }

  context.putImageData(imageData, 0, 0);
}

function getPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;

  return {
    x: ((clientX - rect.left) / rect.width) * canvas.width,
    y: ((clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDrawing(event) {
  drawing = true;
  const point = getPoint(event);
  context.beginPath();
  context.moveTo(point.x, point.y);
  event.preventDefault();
}

function draw(event) {
  if (!drawing) {
    return;
  }

  const point = getPoint(event);
  context.lineTo(point.x, point.y);
  context.stroke();
  event.preventDefault();
}

function stopDrawing() {
  drawing = false;
  context.closePath();
}

function clearCanvas() {
  resizeCanvas();
  renderPlaceholder();
  setModelStatus(
    modelsLoaded ? "Ready. Inference runs entirely in your browser." : "Loading models into your browser...",
    modelsLoaded ? "ready" : "loading",
  );
}

function isCanvasBlank() {
  const palette = getCanvasPalette();
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let index = 0; index < pixels.length; index += 4) {
    if (palette.blankComparator === "lt") {
      if (
        pixels[index] < palette.blankThreshold ||
        pixels[index + 1] < palette.blankThreshold ||
        pixels[index + 2] < palette.blankThreshold
      ) {
        return false;
      }
    } else if (
      pixels[index] > palette.blankThreshold ||
      pixels[index + 1] > palette.blankThreshold ||
      pixels[index + 2] > palette.blankThreshold
    ) {
      return false;
    }
  }
  return true;
}

function getImageForPrediction() {
  const offscreenCanvas = document.createElement("canvas");
  offscreenCanvas.width = canvas.width;
  offscreenCanvas.height = canvas.height;

  const offscreenContext = offscreenCanvas.getContext("2d");
  offscreenContext.drawImage(canvas, 0, 0);

  if (currentTheme !== "dark") {
    return offscreenCanvas;
  }

  const imageData = offscreenContext.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
  const pixels = imageData.data;
  for (let index = 0; index < pixels.length; index += 4) {
    pixels[index] = 255 - pixels[index];
    pixels[index + 1] = 255 - pixels[index + 1];
    pixels[index + 2] = 255 - pixels[index + 2];
  }

  offscreenContext.putImageData(imageData, 0, 0);
  return offscreenCanvas;
}

function renderPlaceholder() {
  resultsGridEl.innerHTML = "";
  const labels = ["CNN", "FCNN"];

  labels.forEach((label) => {
    const placeholder = {
      prediction: "-",
      confidence: 0,
      probabilities: Array.from({ length: 10 }, () => 0),
      isPlaceholder: true,
    };
    resultsGridEl.appendChild(renderModelResult(label, placeholder));
  });
}

function createMetric(labelText, valueText) {
  const block = document.createElement("div");

  const label = document.createElement("p");
  label.className = "label";
  label.textContent = labelText;

  const value = document.createElement("div");
  value.className = "metric-value";
  value.textContent = valueText;

  block.appendChild(label);
  block.appendChild(value);
  return block;
}

function createDistributionChart(probabilities, { placeholder = false } = {}) {
  const chart = document.createElement("div");
  chart.className = "dist-chart";

  probabilities.forEach((probability, digit) => {
    const col = document.createElement("div");
    col.className = "dist-col";

    const pct = document.createElement("div");
    pct.className = "dist-pct";
    pct.textContent = placeholder ? "" : `${(probability * 100).toFixed(0)}%`;

    const barWrap = document.createElement("div");
    barWrap.className = "dist-bar-wrap";

    const bar = document.createElement("div");
    bar.className = "dist-bar";
    bar.style.height = placeholder ? "0%" : `${Math.max(probability * 100, 2)}%`;
    barWrap.appendChild(bar);

    const digitLabel = document.createElement("div");
    digitLabel.className = "dist-digit";
    digitLabel.textContent = String(digit);

    col.appendChild(pct);
    col.appendChild(barWrap);
    col.appendChild(digitLabel);
    chart.appendChild(col);
  });

  return chart;
}

function renderModelResult(modelLabel, result) {
  const card = document.createElement("section");
  card.className = "result-card";

  const header = document.createElement("div");
  header.className = "result-card-header";

  const chip = document.createElement("div");
  chip.className = "model-chip";
  chip.textContent = modelLabel;
  header.appendChild(chip);

  const metrics = document.createElement("div");
  metrics.className = "metric-row";
  metrics.appendChild(createMetric("Prediction", String(result.prediction)));
  metrics.appendChild(
    createMetric(
      "Confidence",
      result.isPlaceholder ? "-" : `${(result.confidence * 100).toFixed(1)}%`,
    ),
  );

  card.appendChild(header);
  card.appendChild(metrics);
  card.appendChild(createDistributionChart(result.probabilities, { placeholder: Boolean(result.isPlaceholder) }));
  return card;
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - maxLogit));
  const sum = exps.reduce((accumulator, value) => accumulator + value, 0);
  return exps.map((value) => value / sum);
}

function buildInvertedImageData(sourceImageData) {
  const { width, height, data } = sourceImageData;
  const invertedPixels = new Uint8ClampedArray(width * height);
  const outImageData = new ImageData(width, height);

  for (let i = 0; i < width * height; i += 1) {
    const dataIndex = i * 4;
    const gray = (data[dataIndex] + data[dataIndex + 1] + data[dataIndex + 2]) / 3;
    const inverted = 255 - gray;

    invertedPixels[i] = inverted;
    outImageData.data[dataIndex] = inverted;
    outImageData.data[dataIndex + 1] = inverted;
    outImageData.data[dataIndex + 2] = inverted;
    outImageData.data[dataIndex + 3] = 255;
  }

  return { invertedPixels, imageData: outImageData };
}

function extractBoundingBox(invertedPixels, width, height) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = invertedPixels[y * width + x];
      if (value <= FOREGROUND_THRESHOLD) {
        continue;
      }

      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  if (maxX === -1 || maxY === -1) {
    return null;
  }

  return {
    minX,
    minY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
  };
}

function getNormalizedTensorFromCanvas(processedCanvas) {
  const pixels = processedCanvas.getContext("2d").getImageData(0, 0, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE).data;
  const tensor = new Float32Array(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE);

  for (let index = 0; index < tensor.length; index += 1) {
    const value = pixels[index * 4] / 255;
    tensor[index] = (value - MNIST_MEAN) / MNIST_STD;
  }

  return tensor;
}

function buildModelInputTensor() {
  const predictionCanvas = getImageForPrediction();
  const sourceCtx = predictionCanvas.getContext("2d");
  const sourceImageData = sourceCtx.getImageData(0, 0, predictionCanvas.width, predictionCanvas.height);
  const { invertedPixels, imageData: invertedImageData } = buildInvertedImageData(sourceImageData);
  const bbox = extractBoundingBox(invertedPixels, predictionCanvas.width, predictionCanvas.height);

  if (!bbox) {
    const blankTensor = new Float32Array(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE);
    blankTensor.fill(NORMALIZED_BLACK);
    return blankTensor;
  }

  const invertedCanvas = document.createElement("canvas");
  invertedCanvas.width = predictionCanvas.width;
  invertedCanvas.height = predictionCanvas.height;
  const invertedCtx = invertedCanvas.getContext("2d");
  invertedCtx.putImageData(invertedImageData, 0, 0);

  const longestSide = Math.max(bbox.width, bbox.height);
  const scale = MNIST_DIGIT_TARGET_SIZE / longestSide;
  const resizedWidth = Math.max(1, Math.round(bbox.width * scale));
  const resizedHeight = Math.max(1, Math.round(bbox.height * scale));

  const digitCanvas = document.createElement("canvas");
  digitCanvas.width = resizedWidth;
  digitCanvas.height = resizedHeight;
  const digitCtx = digitCanvas.getContext("2d");
  digitCtx.imageSmoothingEnabled = true;
  digitCtx.imageSmoothingQuality = "high";
  digitCtx.drawImage(
    invertedCanvas,
    bbox.minX,
    bbox.minY,
    bbox.width,
    bbox.height,
    0,
    0,
    resizedWidth,
    resizedHeight,
  );

  const normalizedCanvas = document.createElement("canvas");
  normalizedCanvas.width = MNIST_IMAGE_SIZE;
  normalizedCanvas.height = MNIST_IMAGE_SIZE;
  const normalizedCtx = normalizedCanvas.getContext("2d");
  normalizedCtx.fillStyle = "black";
  normalizedCtx.fillRect(0, 0, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE);

  const left = Math.floor((MNIST_IMAGE_SIZE - resizedWidth) / 2);
  const top = Math.floor((MNIST_IMAGE_SIZE - resizedHeight) / 2);
  normalizedCtx.drawImage(digitCanvas, left, top);

  return getNormalizedTensorFromCanvas(normalizedCanvas);
}

async function initializeModels() {
  if (!window.ort) {
    setModelStatus("Could not load ONNX Runtime. Refresh to try again.", "error");
    setPredictEnabled(false);
    return;
  }

  if (window.ort.env?.wasm) {
    window.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  }

  setModelStatus("Loading models into your browser...", "loading");
  setPredictEnabled(false);

  try {
    const [cnn, fcnn] = await Promise.all([
      createInferenceSession("cnn"),
      createInferenceSession("fcnn"),
    ]);

    modelSessions = { cnn, fcnn };
    modelsLoaded = true;
    setPredictEnabled(true);
    setModelStatus("Ready. Inference runs entirely in your browser.", "ready");
  } catch (error) {
    console.error(error);
    setModelStatus("Model loading failed. Check console/network and refresh.", "error");
    setPredictEnabled(false);
  }
}

async function runLocalPrediction(modelType, inputData) {
  const session = modelSessions[modelType];
  if (!session) {
    throw new Error(`Model session not loaded: ${modelType}`);
  }

  const inputTensor = new window.ort.Tensor("float32", inputData, [1, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE]);
  const outputs = await session.run({ input: inputTensor });
  const logitsTensor = outputs.logits || Object.values(outputs)[0];

  if (!logitsTensor?.data) {
    throw new Error(`No logits returned from ${modelType} model`);
  }

  const probabilities = softmax(Array.from(logitsTensor.data));
  const prediction = probabilities.reduce(
    (bestIndex, value, index, array) => (value > array[bestIndex] ? index : bestIndex),
    0,
  );

  return {
    prediction,
    confidence: probabilities[prediction],
    probabilities,
  };
}

async function predict() {
  if (predictionInFlight) {
    return;
  }

  if (!modelsLoaded) {
    return;
  }

  if (isCanvasBlank()) {
    return;
  }

  predictionInFlight = true;
  predictBtn.disabled = true;
  setModelStatus("Running inference in your browser...", "loading");

  try {
    const inputTensorData = buildModelInputTensor();
    const cnnResult = await runLocalPrediction("cnn", inputTensorData);
    const fcnnResult = await runLocalPrediction("fcnn", inputTensorData);

    const cnnCard = renderModelResult("CNN", cnnResult);
    const fcnnCard = renderModelResult("FCNN", fcnnResult);
    resultsGridEl.replaceChildren(cnnCard, fcnnCard);
    setModelStatus("Ready. Inference runs entirely in your browser.", "ready");
  } catch (error) {
    console.error(error);
    setModelStatus("Inference failed. Check console for details.", "error");
  } finally {
    predictionInFlight = false;
    setPredictEnabled(modelsLoaded);
  }
}

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);
canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDrawing);

predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", clearCanvas);

if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", () => {
    userOverrodeTheme = true;
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    invertCanvasPixels();
    updateCanvasDrawingStyles();
  });
}

darkSchemeQuery.addEventListener("change", (event) => {
  if (userOverrodeTheme) {
    return;
  }

  setTheme(event.matches ? "dark" : "light");
  invertCanvasPixels();
  updateCanvasDrawingStyles();
});

setTheme(darkSchemeQuery.matches ? "dark" : "light");
resizeCanvas();
renderPlaceholder();
setPredictEnabled(false);
setModelStatus("Loading models into your browser...", "loading");
void initializeModels();