"""
GeoSeer Clone - Two-Stage FastAPI Backend with Google Search Grounding
---------------------------------------------------------------------
1. Stage 1: Uses Google Search tool to dynamically look up Azerbaijani 
   regional landmarks and output highly accurate, grounded reasoning.
2. Stage 2: Formats the verified facts from Stage 1 into the localized Azerbaijani JSON.
"""

import io
import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types
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
    logger.warning("GEMINI_API_KEY is not set. Please set it in backend/.env")
    client = None
else:
    client = genai.Client(api_key=GEMINI_API_KEY)

# Canlı axtarış və detallı analiz üçün gemini-2.5-flash və ya gemini-1.5-pro idealdır
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="GeoSeer Clone API - Grounded Two Stage",
    description="Live-search grounded AI image geolocation",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    confidence: Optional[str] = Field(None, description="Model's self-reported confidence: low, medium, or high")
    explanation: str = Field(..., description="Brief reasoning behind the prediction (2-4 sentences summary)")

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are GeoSeer, an elite OSINT and visual geolocation expert.
Your mission is to find the EXACT geographic location (including country, and especially city, region, or rayon like in Azerbaijan).

Since you have access to Google Search, you MUST perform a live search to verify facts and avoid hallucination:
1. READ INSCRIPTIONS: Carefully identify any text (e.g., "Heydər Əliyev Mərkəzi"). Since almost every rayon in Azerbaijan has a center with this name, do NOT blindly guess Baku, Ganja, or Fuzuli.
2. GOOGLE SEARCH ARCHITECTURE: Use your search tool to look up the specific architectural style of "Heydər Əliyev Mərkəzi" in various regions of Azerbaijan. Match the exact design features (e.g., bronze/copper domes, stone columns, statues, surrounding area) with online reference photos.
3. SEARCH GEOGRAPHICAL CONTEXT: Match the background terrain (mountains, hills, green forests, flatlands) with the topography of Azerbaijani rayons.
4. COMPREHENSIVE EVALUATION: Compare all candidates based on the actual visual evidence and your live web search results.

Summarize your findings step-by-step in English to maintain internal analytical reasoning, ending with a highly verified specific district/city and its exact coordinates."""


FINAL_JSON_INSTRUCTION = """Based ONLY on your search and analysis above, output your final answer as
a single, valid JSON object and NOTHING else. 

CRITICAL RULES:
1. "country" field must be "Azərbaycan" (or the correct country name in Azerbaijani).
2. "city" field MUST NOT be null. It must be the specific rayon, city, or village (e.g., "Gədəbəy", "Gəncə", "İsmayıllı", "Şəmkir") in AZERBAIJANI, verified by your live search.
3. "explanation" field MUST be written in fluent, natural AZERBAIJANI (2-3 sentences explaining exactly how the live search confirmed the visual clues and the exact rayon).

The JSON object must have EXACTLY this shape:
{
  "latitude": <float>,
  "longitude": <float>,
  "country": "<string in Azerbaijani>",
  "city": "<string in Azerbaijani, specific city or rayon>",
  "confidence": "<low, medium, high>",
  "explanation": "<string in Azerbaijani>"
}"""

# --------------------------------------------------------------------------
# Helpers & Routes
# --------------------------------------------------------------------------

def _validate_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return Image.open(io.BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


@app.post("/api/analyze", response_model=GeoLocationResult)
async def analyze_image(file: UploadFile = File(...)) -> GeoLocationResult:
    if not client:
        raise HTTPException(status_code=502, detail="GEMINI_API_KEY is missing on server.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    file_bytes = await file.read()
    pil_image = _validate_image(file_bytes)

    try:
        # === MƏRHƏLƏ 1: DƏRİN ANALİZ (Canlı Google Search Aktivdir) ===
        logger.info("Starting Stage 1: Grounded Reasoning with Google Search...")
        
        stage1_config = types.GenerateContentConfig(
            temperature=0.4,
            # Google GenAI SDK-da rəsmi Google Search toolunun qoşulması
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )

        stage1_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[ANALYSIS_PROMPT, pil_image],
            config=stage1_config
        )
        
        reasoning_text = stage1_response.text
        logger.info("Stage 1 complete. Reasoning with grounding results: %s", reasoning_text)

        # === MƏRHƏLƏ 2: AZƏRBAYCAN DİLİNDƏ STRUKTURLAŞDIRILMIŞ JSON-A ÇEVİRMƏ ===
        logger.info("Starting Stage 2: Formatting to Azerbaijani JSON...")
        
        stage2_config = types.GenerateContentConfig(
            temperature=0.1, 
            response_mime_type="application/json",
            response_schema=GeoLocationResult,
        )

        stage2_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                f"Previous Analysis and verified search data:\n{reasoning_text}\n\nInstruction:\n{FINAL_JSON_INSTRUCTION}"
            ],
            config=stage2_config
        )

        raw_json = stage2_response.text
        logger.info("Stage 2 complete. Raw JSON: %s", raw_json)

        parsed_data = json.loads(raw_json)
        result = GeoLocationResult(**parsed_data)

    except json.JSONDecodeError as exc:
        logger.error("JSON parse error: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to format the final result to JSON.") from exc
    except Exception as exc:
        logger.exception("API process failed")
        raise HTTPException(status_code=502, detail=f"Request failed: {exc}") from exc

    return result


@app.get("/")
async def root():
    return {"message": "Grounded Two-Stage GeoSeer API is live."}