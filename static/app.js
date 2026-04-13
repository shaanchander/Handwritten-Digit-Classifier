const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const modelTypeSelect = document.getElementById("modelType");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const statusEl = document.getElementById("status");
const barsEl = document.getElementById("bars");

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
  predictionEl.textContent = "-";
  confidenceEl.textContent = "-";
  statusEl.textContent = "Ready.";
  barsEl.innerHTML = "";
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

function renderBars(probabilities) {
  barsEl.innerHTML = "";

  probabilities.forEach((value, digit) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("div");
    label.textContent = digit;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(value * 100, 1)}%`;
    track.appendChild(fill);

    const amount = document.createElement("div");
    amount.className = "bar-value";
    amount.textContent = `${(value * 100).toFixed(1)}%`;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(amount);
    barsEl.appendChild(row);
  });
}

async function predict() {
  if (isCanvasBlank()) {
    statusEl.textContent = "Draw a digit before predicting.";
    return;
  }

  predictBtn.disabled = true;
  statusEl.textContent = "Running inference...";

  try {
    const response = await fetch(`/predict/${modelTypeSelect.value}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") }),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Prediction failed");
    }

    predictionEl.textContent = result.prediction;
    confidenceEl.textContent = `${(result.confidence * 100).toFixed(1)}%`;
    statusEl.textContent = "Prediction complete.";
    renderBars(result.probabilities);
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