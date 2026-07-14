# GeoSeer Clone

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
source venv/bin/activate      # on Windows: venv\Scripts\activate

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
