const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyze-btn");
const resultText = document.getElementById("result-text");
const confidenceText = document.getElementById("confidence-text");
const statusPill = document.getElementById("status-pill");
const fileName = document.getElementById("file-name");
const dropZone = document.getElementById("drop-zone");

let selectedFile = null;

const setStatus = (label, tone) => {
  statusPill.textContent = label;
  statusPill.classList.remove("success", "danger");
  if (tone) {
    statusPill.classList.add(tone);
  }
};

const setResult = (label, confidence, tone) => {
  resultText.textContent = `Prediction: ${label}`;
  confidenceText.textContent = `Confidence: ${confidence}`;
  setStatus("Ready", tone);
};

const handleFile = (file) => {
  if (!file) {
    return;
  }

  selectedFile = file;
  fileName.textContent = file.name;
  analyzeBtn.disabled = false;

  const objectUrl = URL.createObjectURL(file);
  preview.src = objectUrl;
  preview.classList.remove("is-hidden");
  setStatus("Loaded", null);
  resultText.textContent = "Prediction: --";
  confidenceText.textContent = "Confidence: --";
};

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  handleFile(file);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const [file] = event.dataTransfer.files;
  handleFile(file);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  analyzeBtn.disabled = true;
  setStatus("Analyzing", null);
  resultText.textContent = "Prediction: Running...";
  confidenceText.textContent = "Confidence: --";

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const data = await response.json();
    const label = data.label ? data.label.toUpperCase() : "UNKNOWN";
    const confidenceValue =
      typeof data.confidence === "number"
        ? `${(data.confidence * 100).toFixed(1)}%`
        : "--";

    const tone = label === "DEEPFAKE" ? "danger" : "success";
    setResult(label, confidenceValue, tone);
  } catch (error) {
    setStatus("Error", "danger");
    resultText.textContent = "Prediction: Failed";
    confidenceText.textContent = "Confidence: --";
  } finally {
    analyzeBtn.disabled = false;
  }
});
