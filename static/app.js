const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const modelTypeSelect = document.getElementById("modelType");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const resultsGridEl = document.getElementById("resultsGrid");
const themeToggleBtn = document.getElementById("theme-toggle") || document.getElementById("themeToggle");
const darkSchemeQuery = window.matchMedia("(prefers-color-scheme: dark)");

let drawing = false;
let currentTheme = "light";
let userOverrodeTheme = false;

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
  statusEl.textContent = "Ready.";
  renderPlaceholder();
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
  if (currentTheme !== "dark") {
    return canvas.toDataURL("image/png");
  }

  const offscreenCanvas = document.createElement("canvas");
  offscreenCanvas.width = canvas.width;
  offscreenCanvas.height = canvas.height;

  const offscreenContext = offscreenCanvas.getContext("2d");
  offscreenContext.drawImage(canvas, 0, 0);

  const imageData = offscreenContext.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
  const pixels = imageData.data;
  for (let index = 0; index < pixels.length; index += 4) {
    pixels[index] = 255 - pixels[index];
    pixels[index + 1] = 255 - pixels[index + 1];
    pixels[index + 2] = 255 - pixels[index + 2];
  }

  offscreenContext.putImageData(imageData, 0, 0);
  return offscreenCanvas.toDataURL("image/png");
}

function renderPlaceholder() {
  resultsGridEl.innerHTML = "";
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

function createDistributionChart(probabilities) {
  const chart = document.createElement("div");
  chart.className = "dist-chart";

  probabilities.forEach((probability, digit) => {
    const col = document.createElement("div");
    col.className = "dist-col";

    const pct = document.createElement("div");
    pct.className = "dist-pct";
    pct.textContent = `${(probability * 100).toFixed(0)}%`;

    const barWrap = document.createElement("div");
    barWrap.className = "dist-bar-wrap";

    const bar = document.createElement("div");
    bar.className = "dist-bar";
    bar.style.height = `${Math.max(probability * 100, 2)}%`;
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
  metrics.appendChild(createMetric("Confidence", `${(result.confidence * 100).toFixed(1)}%`));

  card.appendChild(header);
  card.appendChild(metrics);
  card.appendChild(createDistributionChart(result.probabilities));
  return card;
}

async function requestPrediction(modelType, imageData) {
  const response = await fetch(`/predict/${modelType}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image: imageData }),
  });

  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.detail || "Prediction failed");
  }
  return result;
}

async function predict() {
  if (isCanvasBlank()) {
    statusEl.textContent = "Draw a digit before predicting.";
    return;
  }

  predictBtn.disabled = true;
  statusEl.textContent = "Running inference...";

  try {
    const imageData = getImageForPrediction();
    const modelType = modelTypeSelect.value;
    resultsGridEl.innerHTML = "";

    if (modelType === "both") {
      const [cnnResult, fcnnResult] = await Promise.all([
        requestPrediction("cnn", imageData),
        requestPrediction("fcnn", imageData),
      ]);
      resultsGridEl.appendChild(renderModelResult("CNN", cnnResult));
      resultsGridEl.appendChild(renderModelResult("FCNN", fcnnResult));
    } else {
      const result = await requestPrediction(modelType, imageData);
      const label = modelType === "cnn" ? "CNN" : "FCNN";
      resultsGridEl.appendChild(renderModelResult(label, result));
    }
    statusEl.textContent = "Ready.";
  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    predictBtn.disabled = false;
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