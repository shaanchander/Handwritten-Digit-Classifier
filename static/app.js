const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const modelTypeSelect = document.getElementById("modelType");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const resultsGridEl = document.getElementById("resultsGrid");

let drawing = false;

function resizeCanvas() {
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "#111111";
  context.lineWidth = 24;
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
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let index = 0; index < pixels.length; index += 4) {
    if (pixels[index] < 250 || pixels[index + 1] < 250 || pixels[index + 2] < 250) {
      return false;
    }
  }
  return true;
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
    const imageData = canvas.toDataURL("image/png");
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

resizeCanvas();
renderPlaceholder();