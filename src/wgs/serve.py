"""HTTP wrapper around wgs.service for the local demo.

Run:  python -m uvicorn wgs.serve:app --host 127.0.0.1 --port 8000
Config: WGS_CONFIG env var (default configs/www_short.yaml). Requires the `serve` extra:
        pip install -e ".[serve]"
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import load_config
from .service import analyze, parse_reviews

app = FastAPI(title="WhoGetsSummarized local analysis", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://haeinseo.github.io", "http://127.0.0.1:8765", "http://localhost:8765"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def _cfg():
    return load_config(os.environ.get("WGS_CONFIG", "configs/www_short.yaml"))


class AnalyzeIn(BaseModel):
    reviews_text: str = Field(..., description="One review per line or blank-line-separated block")
    category: str = "product"


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/analyze")
def analyze_endpoint(body: AnalyzeIn) -> dict:
    reviews = parse_reviews(body.reviews_text)
    if len(reviews) < 3:
        raise HTTPException(status_code=422, detail="Paste at least 3 reviews (20+ characters each).")
    try:
        return analyze(_cfg(), reviews, category=body.category)
    except Exception as e:  # surface the reason instead of a bare 500
        raise HTTPException(status_code=503, detail=f"analysis failed: {e}") from e
