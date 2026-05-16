"""
config.py — Central configuration for Lingo Speech Pipeline
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "Lingo Speech Pipeline"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    OUTPUT_DIR: Path = DATA_DIR / "outputs"

    # ── Whisper ───────────────────────────────────────────────────────────────
    WHISPER_MODEL: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"] = "base"
    WHISPER_DEVICE: str = "cpu"          # "cuda" if GPU available
    WHISPER_COMPUTE_TYPE: str = "int8"   # "float16" on GPU
    WHISPER_LANGUAGE: str | None = "ta"  # Tamil default; None = auto-detect
    WHISPER_BATCH_SIZE: int = 8

    # ── Diarization ───────────────────────────────────────────────────────────
    PYANNOTE_TOKEN: str = ""             # HuggingFace token for pyannote
    MIN_SPEAKERS: int = 1
    MAX_SPEAKERS: int = 10

    # ── Emotion ───────────────────────────────────────────────────────────────
    EMOTION_MODEL: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    EMOTION_LABELS: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    # ── Audio ─────────────────────────────────────────────────────────────────
    TARGET_SAMPLE_RATE: int = 16000
    TARGET_CHANNELS: int = 1            # mono
    MAX_AUDIO_DURATION_SECS: int = 3600 # 1 hour
    MIN_SEGMENT_DURATION: float = 0.5   # seconds
    CHUNK_DURATION_SECS: int = 30       # for chunked processing

    # ── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    MAX_UPLOAD_SIZE_MB: int = 500
    CORS_ORIGINS: list[str] = ["*"]

    # ── Redis / Celery ────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER: str = "redis://localhost:6379/0"
    CELERY_BACKEND: str = "redis://localhost:6379/1"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "pipeline.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
for d in [settings.RAW_DIR, settings.PROCESSED_DIR, settings.OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
