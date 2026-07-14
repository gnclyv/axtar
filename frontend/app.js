// GeoSeer Clone - Frontend Logic
// --------------------------------
// Handles image selection/drag-and-drop, calls the FastAPI backend,
// and renders the AI's geolocation prediction on a Leaflet map.

// Əgər lokal fayl kimi açılıbsa belə, mütləq localhost:8000 portuna yönləndiririk
const API_BASE_URL = window.location.origin;
const ANALYZE_ENDPOINT = `${API_BASE_URL}/api/analyze`;

// --- DOM references -------------------------------------------------------

const dropZone = document.getElementById("drop-zone");
const dropZonePlaceholder = document.getElementById("drop-zone-placeholder");
const fileInput = document.getElementById("file-input");
const previewImage = document.getElementById("preview-image");

const analyzeBtn = document.getElementById("analyze-btn");
const analyzeBtnText = document.getElementById("analyze-btn-text");
const analyzeSpinner = document.getElementById("analyze-spinner");

const errorBox = document.getElementById("error-box");

const resultsCard = document.getElementById("results-card");
const resultCountry = document.getElementById("result-country");
const resultCity = document.getElementById("result-city");
const resultCoords = document.getElementById("result-coords");
const resultConfidence = document.getElementById("result-confidence");
const resultExplanation = document.getElementById("result-explanation");

// --- State ------------------------------------------------------------

let selectedFile = null;
let map = null;
let marker = null;

// --- Map initialization -------------------------------------------------

function initMap() {
  map = L.map("map", {
    zoomControl: true,
    attributionControl: true,
  }).setView([20, 0], 2);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
}

function updateMap(lat, lng, popupHtml) {
  if (!map) return;

  if (marker) {
    map.removeLayer(marker);
  }

  marker = L.marker([lat, lng]).addTo(map);
  if (popupHtml) {
    marker.bindPopup(popupHtml).openPopup();
  }

  map.flyTo([lat, lng], 6, { duration: 1.5 });
}

// --- UI helpers -------------------------------------------------------

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading || !selectedFile;
  analyzeSpinner.classList.toggle("hidden", !isLoading);
  analyzeBtnText.textContent = isLoading ? "Analyzing..." : "Locate this image";
}

function handleFileSelected(file) {
  if (!file) return;

  const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
  if (!allowedTypes.includes(file.type)) {
    showError("Unsupported file type. Please upload a JPEG, PNG, or WEBP image.");
    return;
  }

  const maxBytes = 15 * 1024 * 1024;
  if (file.size > maxBytes) {
    showError("File is too large. Maximum size is 15MB.");
    return;
  }

  clearError();
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.classList.remove("hidden");
    dropZonePlaceholder.classList.add("hidden");
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
}

function displayResults(data) {
  resultCountry.textContent = data.country || "Unknown";
  resultCity.textContent = data.city || "Unknown";
  resultCoords.textContent = `${data.latitude.toFixed(5)}, ${data.longitude.toFixed(5)}`;
  resultConfidence.textContent = data.confidence || "medium";
  resultExplanation.textContent = data.explanation || "No explanation provided.";

  resultsCard.classList.remove("hidden");

  const popupHtml = `
    <div style="font-family: sans-serif; max-width: 220px;">
      <strong>${data.city ? data.city + ", " : ""}${data.country}</strong>
      <p style="margin-top: 4px; font-size: 12px;">${data.explanation}</p>
    </div>
  `;

  updateMap(data.latitude, data.longitude, popupHtml);
}

// --- API call -----------------------------------------------------------

async function analyzeImage() {
  if (!selectedFile) return;

  clearError();
  setLoading(true);

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(ANALYZE_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `Request failed with status ${response.status}`);
    }

    displayResults(data);
  } catch (err) {
    console.error("Analysis failed:", err);
    showError(err.message || "Something went wrong while analyzing the image. Please try again.");
  } finally {
    setLoading(false);
  }
}

// --- Event listeners ------------------------------------------------------

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  handleFileSelected(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("drop-zone-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("drop-zone-active");
  });
});

dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  handleFileSelected(file);
});

analyzeBtn.addEventListener("click", analyzeImage);

// --- Init -----------------------------------------------------------------

document.addEventListener("DOMContentLoaded", initMap);
