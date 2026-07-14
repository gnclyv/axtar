#!/usr/bin/env python3
"""
generate_project.py

Scaffolds a complete "GeoSeer" clone project:
  - backend/main.py        (FastAPI + Gemini Pro Vision integration)
  - frontend/index.html    (Tailwind CSS UI + Leaflet map container)
  - frontend/app.js        (Upload logic, API calls, map rendering)
  - requirements.txt
  - README.md

Run this script once:
    python generate_project.py

It will create ./backend and ./frontend directories with all files,
then bundle everything into "geoseer_clone.zip" in the current directory.
"""

import os
import zipfile
from pathlib import Path

ROOT = Path.cwd()
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
ZIP_NAME = ROOT / "geoseer_clone.zip"


# --------------------------------------------------------------------------
# FILE CONTENTS
# --------------------------------------------------------------------------

MAIN_PY = '''"""
GeoSeer Clone - FastAPI Backend
--------------------------------
Accepts an uploaded image, sends it to Google's Gemini Pro Vision model,
and returns a structured JSON payload with the model's best guess at the
image's geographic location (latitude, longitude, country, city) along
with a short explanation of the visual clues it used.
"""

import io
import json
import logging
import os
import re
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("geoseer")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is not set. Create a .env file in the backend/ "
        "directory (or export the variable) before making requests."
    )
else:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}

app = FastAPI(
    title="GeoSeer Clone API",
    description="AI-powered image geolocation using Gemini Pro Vision",
    version="1.0.0",
)

# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production to your frontend origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class GeoLocationResult(BaseModel):
    latitude: float = Field(..., description="Predicted latitude in decimal degrees")
    longitude: float = Field(..., description="Predicted longitude in decimal degrees")
    country: str = Field(..., description="Predicted country")
    city: Optional[str] = Field(None, description="Predicted city or nearest locality")
    confidence: Optional[str] = Field(
        None, description="Model's self-reported confidence: low, medium, or high"
    )
    explanation: str = Field(..., description="Brief reasoning behind the prediction")


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are GeoSeer, a world-class visual geolocation expert, similar in
spirit to GeoGuessr champions. You will be shown a single photograph. Your job is to
determine, as precisely as possible, WHERE on Earth this photo was taken.

Analyze every available visual clue, including but not limited to:
- Architecture, building materials, and construction styles
- Road markings, signage, license plates, and traffic signals
- Vegetation, terrain, climate, and geology
- Language(s) visible in any text
- Utility poles, infrastructure design, and construction codes
- Sun position/shadows (hemisphere and rough time of day)
- Cultural cues: clothing, vehicles, shop fronts, flags
- Soil color, mountains, coastlines, or other geographic landmarks

Think step by step internally, weighing multiple hypotheses, then converge on your single
best final answer.

You MUST respond with ONLY a single valid JSON object and nothing else - no markdown code
fences, no preamble, no explanation outside the JSON. The JSON object must have EXACTLY
this shape:

{
  "latitude": <float, decimal degrees, required>,
  "longitude": <float, decimal degrees, required>,
  "country": "<string, best-guess country name, required>",
  "city": "<string, best-guess city or nearest locality, or null if unknown>",
  "confidence": "<one of: low, medium, high>",
  "explanation": "<2-4 sentences describing the specific visual clues that led to this conclusion>"
}

If you are genuinely uncertain, still provide your single best-guess coordinates (never
leave latitude/longitude null) and reflect your uncertainty honestly in the "confidence"
and "explanation" fields."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _extract_json(raw_text: str) -> dict:
    """Extract and parse a JSON object from the model's raw text response,
    tolerating markdown code fences or stray whitespace/text around it."""
    text = raw_text.strip()

    # Strip markdown code fences like ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Fall back to grabbing the first {...} block in the text
        brace_match = re.search(r"\\{.*\\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    return json.loads(text)


def _validate_image(file_bytes: bytes) -> None:
    """Raise HTTPException if the bytes are not a valid, readable image."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", gemini_configured=bool(GEMINI_API_KEY))


@app.post("/api/analyze", response_model=GeoLocationResult)
async def analyze_image(file: UploadFile = File(...)) -> GeoLocationResult:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server is missing GEMINI_API_KEY. Set it in backend/.env and restart the server.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    _validate_image(file_bytes)

    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)

        image_part = {"mime_type": file.content_type, "data": file_bytes}

        generation_config = genai.types.GenerationConfig(
            temperature=0.4,
            top_p=0.95,
            max_output_tokens=1024,
        )

        response = model.generate_content(
            [SYSTEM_PROMPT, image_part],
            generation_config=generation_config,
        )

        raw_text = response.text if hasattr(response, "text") else str(response)
        logger.info("Raw Gemini response: %s", raw_text)

        parsed = _extract_json(raw_text)

    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Gemini response as JSON: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The AI model returned a response that could not be parsed. Please try again.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemini API call failed")
        raise HTTPException(status_code=502, detail=f"AI model request failed: {exc}") from exc

    try:
        result = GeoLocationResult(
            latitude=float(parsed["latitude"]),
            longitude=float(parsed["longitude"]),
            country=str(parsed.get("country", "Unknown")),
            city=parsed.get("city"),
            confidence=parsed.get("confidence", "medium"),
            explanation=str(parsed.get("explanation", "")),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("Gemini response missing expected fields: %s | parsed=%s", exc, parsed)
        raise HTTPException(
            status_code=502,
            detail="The AI model's response was missing required geolocation fields.",
        ) from exc

    return result


@app.get("/")
async def root():
    return {
        "message": "GeoSeer Clone API is running.",
        "docs": "/docs",
        "health": "/api/health",
        "analyze_endpoint": "POST /api/analyze (multipart/form-data, field name: file)",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
'''


INDEX_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GeoSeer Clone | AI Image Geolocation</title>

  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>

  <!-- Leaflet CSS -->
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />

  <style>
    html, body { height: 100%; }
    #map { height: 100%; width: 100%; }
    .drop-zone-active {
      border-color: #6366f1 !important;
      background-color: rgba(99, 102, 241, 0.08) !important;
    }
    .spinner {
      border: 3px solid rgba(255, 255, 255, 0.2);
      border-top-color: #6366f1;
      border-radius: 50%;
      width: 22px;
      height: 22px;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 font-sans antialiased">

  <div class="min-h-screen flex flex-col">

    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-30">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-lg">
            🌍
          </div>
          <div>
            <h1 class="text-lg font-semibold tracking-tight">GeoSeer <span class="text-indigo-400">Clone</span></h1>
            <p class="text-xs text-slate-400 -mt-0.5">AI-powered image geolocation</p>
          </div>
        </div>
        <a
          href="https://ai.google.dev/"
          target="_blank"
          rel="noopener noreferrer"
          class="text-xs text-slate-400 hover:text-indigo-400 transition-colors hidden sm:block"
        >
          Powered by Gemini Pro Vision
        </a>
      </div>
    </header>

    <!-- Main content -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 grid grid-cols-1 lg:grid-cols-5 gap-6">

      <!-- Left column: Upload + Results -->
      <section class="lg:col-span-2 flex flex-col gap-5">

        <!-- Upload card -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <h2 class="font-semibold mb-3 text-slate-200">1. Upload an image</h2>

          <div
            id="drop-zone"
            class="relative border-2 border-dashed border-slate-700 rounded-xl p-6 text-center cursor-pointer transition-colors hover:border-indigo-500 hover:bg-indigo-500/5"
          >
            <input id="file-input" type="file" accept="image/png, image/jpeg, image/webp" class="hidden" />

            <div id="drop-zone-placeholder" class="flex flex-col items-center gap-2 py-6">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
              </svg>
              <p class="text-sm text-slate-300 font-medium">Drag & drop an image here</p>
              <p class="text-xs text-slate-500">or click to browse (JPEG, PNG, WEBP, max 15MB)</p>
            </div>

            <img id="preview-image" class="hidden mx-auto max-h-64 rounded-lg object-contain" alt="Preview" />
          </div>

          <button
            id="analyze-btn"
            disabled
            class="mt-4 w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-medium py-2.5 rounded-lg transition-colors"
          >
            <span id="analyze-btn-text">Locate this image</span>
            <span id="analyze-spinner" class="hidden spinner"></span>
          </button>

          <p id="error-box" class="hidden mt-3 text-sm text-red-400 bg-red-950/40 border border-red-900 rounded-lg px-3 py-2"></p>
        </div>

        <!-- Results card -->
        <div id="results-card" class="hidden bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex-1">
          <h2 class="font-semibold mb-4 text-slate-200">2. Prediction</h2>

          <div class="grid grid-cols-2 gap-3 mb-4">
            <div class="bg-slate-800/60 rounded-lg p-3">
              <p class="text-xs text-slate-400 mb-1">Country</p>
              <p id="result-country" class="font-medium text-slate-100">—</p>
            </div>
            <div class="bg-slate-800/60 rounded-lg p-3">
              <p class="text-xs text-slate-400 mb-1">City / Locality</p>
              <p id="result-city" class="font-medium text-slate-100">—</p>
            </div>
            <div class="bg-slate-800/60 rounded-lg p-3">
              <p class="text-xs text-slate-400 mb-1">Coordinates</p>
              <p id="result-coords" class="font-medium text-slate-100 text-sm">—</p>
            </div>
            <div class="bg-slate-800/60 rounded-lg p-3">
              <p class="text-xs text-slate-400 mb-1">Confidence</p>
              <p id="result-confidence" class="font-medium text-slate-100 capitalize">—</p>
            </div>
          </div>

          <div>
            <p class="text-xs text-slate-400 mb-1">AI Reasoning</p>
            <p id="result-explanation" class="text-sm text-slate-300 leading-relaxed">—</p>
          </div>
        </div>
      </section>

      <!-- Right column: Map -->
      <section class="lg:col-span-3 bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl min-h-[400px] lg:min-h-0">
        <div id="map"></div>
      </section>

    </main>

    <footer class="border-t border-slate-800 py-4 text-center text-xs text-slate-500">
      GeoSeer Clone &mdash; for educational/demo purposes only. Not affiliated with the original GeoSeer.
    </footer>
  </div>

  <!-- Leaflet JS -->
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>

  <script src="app.js"></script>
</body>
</html>
'''


APP_JS = '''// GeoSeer Clone - Frontend Logic
// --------------------------------
// Handles image selection/drag-and-drop, calls the FastAPI backend,
// and renders the AI's geolocation prediction on a Leaflet map.

const API_BASE_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://localhost:8000"
  : ""; // In production, set this to your deployed backend URL.

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
'''


REQUIREMENTS_TXT = '''fastapi==0.115.0
uvicorn[standard]==0.30.6
google-generativeai==0.8.3
python-multipart==0.0.9
Pillow==10.4.0
python-dotenv==1.0.1
pydantic==2.9.2
'''


README_MD = '''# GeoSeer Clone

An AI-powered image geolocation web app. Upload a photo, and Gemini Pro Vision will
analyze visual clues (architecture, signage, vegetation, terrain, etc.) to predict
where in the world it was taken -- displaying the result on an interactive Leaflet map.

## Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** HTML + Tailwind CSS (via CDN) + vanilla JS + Leaflet.js
- **AI:** Google Gemini Pro Vision (`google-generativeai`)

## Project Structure

```
.
├── backend/
│   └── main.py
├── frontend/
│   ├── index.html
│   └── app.js
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.9+
- A Google Gemini API key: https://aistudio.google.com/app/apikey

## 1. Backend Setup

```bash
# From the project root
cd backend

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\\Scripts\\activate

# Install dependencies (requirements.txt lives one level up)
pip install -r ../requirements.txt
```

Create a `.env` file inside the `backend/` directory with your API key:

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

Run the FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive API docs: `http://localhost:8000/docs`

## 2. Frontend Setup

The frontend is a static site (no build step required). From the project root:

```bash
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500` in your browser.

> Alternatively, you can just open `frontend/index.html` directly in your browser,
> or use a tool like VS Code's "Live Server" extension.

By default, `app.js` points to `http://localhost:8000` for the backend API when
running locally. Update the `API_BASE_URL` constant in `frontend/app.js` if you
deploy the backend elsewhere.

## 3. Usage

1. Open the frontend in your browser.
2. Drag and drop an image (or click to browse) into the upload area.
3. Click **"Locate this image"**.
4. View the predicted country, city, coordinates, confidence, and AI reasoning,
   plus a marker on the map showing the predicted location.

## Notes & Limitations

- This is a demo/educational project and is **not affiliated with the original
  GeoSeer** product.
- Predictions are AI-generated best guesses and may be inaccurate, especially for
  images with few distinguishing visual clues (e.g., indoor photos, close-ups).
- CORS is wide open (`allow_origins=["*"]`) for local development convenience.
  Restrict this in `backend/main.py` before deploying to production.
- Uploaded images are sent directly to the Gemini API for analysis and are not
  persisted to disk by the backend.
- Max upload size is 15MB; supported formats are JPEG, PNG, and WEBP.

## Troubleshooting

- **"Server is missing GEMINI_API_KEY"** — Make sure `backend/.env` exists and
  contains a valid `GEMINI_API_KEY`, then restart `uvicorn`.
- **CORS errors in the browser console** — Confirm the backend is running on
  `http://localhost:8000` and that `API_BASE_URL` in `app.js` matches.
- **"The AI model returned a response that could not be parsed"** — Occasionally
  the model may not return strict JSON. Simply retry the request.
'''


# --------------------------------------------------------------------------
# Script logic
# --------------------------------------------------------------------------


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  ✔ wrote {path.relative_to(ROOT)}")


def create_project() -> None:
    print("Creating project structure...")
    print()

    print("[backend/]")
    write_file(BACKEND_DIR / "main.py", MAIN_PY)

    print()
    print("[frontend/]")
    write_file(FRONTEND_DIR / "index.html", INDEX_HTML)
    write_file(FRONTEND_DIR / "app.js", APP_JS)

    print()
    print("[root]")
    write_file(ROOT / "requirements.txt", REQUIREMENTS_TXT)
    write_file(ROOT / "README.md", README_MD)


def zip_project() -> None:
    print()
    print(f"Compressing project into {ZIP_NAME.name} ...")

    files_to_zip = [
        BACKEND_DIR / "main.py",
        FRONTEND_DIR / "index.html",
        FRONTEND_DIR / "app.js",
        ROOT / "requirements.txt",
        ROOT / "README.md",
    ]

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_zip:
            arcname = file_path.relative_to(ROOT)
            zf.write(file_path, arcname=arcname)
            print(f"  \u2714 added {arcname}")

    print()
    print(f"Done! Created {ZIP_NAME.name} ({ZIP_NAME.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    create_project()
    zip_project()
    print()
    print("Next steps:")
    print("  1. cd backend && pip install -r ../requirements.txt")
    print("  2. Create backend/.env with GEMINI_API_KEY=your_key_here")
    print("  3. uvicorn main:app --reload --port 8000  (run from inside backend/)")
    print("  4. cd ../frontend && python -m http.server 5500")
    print("  5. Open http://localhost:5500 in your browser")


if __name__ == "__main__":
    main()
