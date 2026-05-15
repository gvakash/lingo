"""
main.py — FastAPI application for Lingo Speech Pipeline
Endpoints: upload, transcribe, diarize, emotion, full pipeline, health
"""
import os
import json
import asyncio
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

from app.config import settings
from app.core.pipeline import SpeechPipeline, PipelineConfig

# ── Global pipeline singleton ──────────────────────────────────────────────────
pipeline: Optional[SpeechPipeline] = None
jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting Lingo Speech Pipeline API...")
    pipeline = SpeechPipeline()
    logger.success("✓ Pipeline initialized and ready")
    yield
    logger.info("Shutting down...")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lingo Speech Pipeline API",
    description=(
        "End-to-end speech transcription, speaker diarization, and emotion tagging pipeline. "
        "Supports 99+ languages including all major Indian languages."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    frontend = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend.exists():
        return HTMLResponse(frontend.read_text())
    return HTMLResponse("""
    <html><body>
    <h1>🎙️ Lingo Speech Pipeline</h1>
    <p>API is running. See <a href="/docs">/docs</a> for the interactive API.</p>
    </body></html>
    """)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "whisper_model": settings.WHISPER_MODEL,
        "device": settings.WHISPER_DEVICE,
        "version": settings.APP_VERSION,
    }


@app.post("/api/pipeline", summary="Full Pipeline: Transcribe + Diarize + Emotion")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (WAV/MP3/M4A/FLAC)"),
    language: Optional[str] = Query(None, description="Language code e.g. 'ta', 'hi', 'en'"),
    run_diarization: bool = Query(True),
    run_emotion: bool = Query(True),
    denoise: bool = Query(True),
    num_speakers: Optional[int] = Query(None),
):
    """
    Run the complete speech pipeline:
    1. Audio preprocessing (denoising, normalization, resampling)
    2. Whisper transcription (with word timestamps)
    3. Speaker diarization (who spoke when)
    4. Emotion & style tagging (per segment)
    """
    _validate_upload(file)
    audio_bytes = await file.read()

    config = PipelineConfig(
        language=language,
        run_diarization=run_diarization,
        run_emotion=run_emotion,
        denoise=denoise,
        num_speakers=num_speakers,
    )

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: pipeline.run_from_bytes(audio_bytes, file.filename, config)
    )

    return JSONResponse({
        "job_id": result.job_id,
        "status": result.status,
        "input_file": result.input_file,
        "audio": result.audio_metadata,
        "transcription": result.transcription,
        "diarization": result.diarization,
        "emotion": result.emotion,
        "output_files": result.output_files,
        "timing": result.stage_times,
        "total_time_secs": result.total_time_secs,
        "errors": result.errors,
    })


@app.post("/api/transcribe", summary="Transcribe Only (Fastest)")
async def transcribe_only(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None),
    word_timestamps: bool = Query(True),
    reference_text: Optional[str] = Query(None, description="For WER/CER evaluation"),
):
    """Fast transcription only — no diarization or emotion tagging."""
    _validate_upload(file)
    audio_bytes = await file.read()

    config = PipelineConfig(
        language=language,
        run_diarization=False,
        run_emotion=False,
        word_timestamps=word_timestamps,
        reference_text=reference_text,
    )

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: pipeline.run_from_bytes(audio_bytes, file.filename, config)
    )

    return JSONResponse({
        "job_id": result.job_id,
        "status": result.status,
        "transcription": result.transcription,
        "audio": result.audio_metadata,
        "timing": result.stage_times,
    })


@app.post("/api/diarize", summary="Diarization Only")
async def diarize_only(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Query(None),
):
    """Speaker diarization — returns RTTM-style speaker turns."""
    _validate_upload(file)
    audio_bytes = await file.read()

    config = PipelineConfig(
        run_diarization=True,
        run_emotion=False,
        num_speakers=num_speakers,
    )

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: pipeline.run_from_bytes(audio_bytes, file.filename, config)
    )

    return JSONResponse({
        "job_id": result.job_id,
        "status": result.status,
        "diarization": result.diarization,
        "timing": result.stage_times,
    })


@app.post("/api/emotion", summary="Emotion Tagging Only")
async def emotion_only(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None),
):
    """Emotion & speaking style tagging per speech segment."""
    _validate_upload(file)
    audio_bytes = await file.read()

    config = PipelineConfig(
        language=language,
        run_diarization=False,
        run_emotion=True,
    )

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: pipeline.run_from_bytes(audio_bytes, file.filename, config)
    )

    return JSONResponse({
        "job_id": result.job_id,
        "status": result.status,
        "emotion": result.emotion,
        "transcription": {
            "segments": (result.transcription or {}).get("segments", [])
        },
        "timing": result.stage_times,
    })


@app.get("/api/outputs/{job_id}/{filename}", summary="Download Output File")
async def download_output(job_id: str, filename: str):
    path = settings.OUTPUT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(str(path), filename=filename)


@app.get("/api/models", summary="List Available Models")
async def list_models():
    return {
        "whisper": {
            "model": settings.WHISPER_MODEL,
            "device": settings.WHISPER_DEVICE,
            "supported_languages": "99+ (auto-detect supported)",
        },
        "diarization": {
            "model": "pyannote/speaker-diarization-3.1",
            "available": pipeline.diarizer._available if pipeline else False,
            "fallback": "energy-based (single speaker)",
        },
        "emotion": {
            "model": settings.EMOTION_MODEL,
            "available": pipeline.emotion_tagger._available if pipeline else False,
            "labels": settings.EMOTION_LABELS,
            "fallback": "prosody-based heuristics",
        },
    }


@app.get("/api/languages", summary="Supported Languages")
async def supported_languages():
    indian_languages = {
        "hi": "Hindi", "bn": "Bengali", "te": "Telugu", "mr": "Marathi",
        "ta": "Tamil", "ur": "Urdu", "gu": "Gujarati", "kn": "Kannada",
        "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
    }
    return {
        "total_supported": 99,
        "auto_detect": True,
        "indian_languages": indian_languages,
        "note": "Set language=None for automatic detection",
    }


def _validate_upload(file: UploadFile):
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
