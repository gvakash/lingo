# 🎙️ Lingo Speech Pipeline

> End-to-end speech transcription, speaker diarization, and emotion tagging — built for real-world multilingual audio.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-orange?style=flat-square)](https://github.com/openai/whisper)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## What This Does

Takes raw audio and outputs structured annotations — transcripts, speaker turns, and emotion labels — ready for downstream ML training pipelines.

```
Audio File (any format)
        │
        ▼
┌───────────────────┐
│  Audio Processing  │  → denoising · resampling · normalization · chunking
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Whisper ASR       │  → full transcript · word timestamps · language detection
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Speaker Diarize   │  → who spoke when · RTTM output · speaker turns
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Emotion Tagging   │  → emotion per segment · speaking style · pitch/energy
└────────┬──────────┘
         │
         ▼
   JSON · RTTM · CSV
```

---

## Features

- **Multi-format audio ingestion** — WAV, MP3, M4A, FLAC, OGG, OPUS with FFmpeg fallback
- **Noise reduction** — statistical noise profiling via `noisereduce`
- **Whisper ASR** — 99 languages, word-level timestamps, WER/CER evaluation
- **Tamil-first** — optimized defaults for Tamil (`ta`), with full support for Hindi, Bengali, Telugu, Kannada, Malayalam, Marathi, Gujarati and 90+ more
- **Speaker diarization** — pyannote.audio 3.1 with energy-based fallback
- **Emotion tagging** — wav2vec2 per segment: angry, happy, sad, neutral, fear, surprise, disgust
- **Speaking style inference** — maps emotion + energy to expressive style labels
- **Batch processing** — process entire directories unattended
- **FastAPI REST API** — upload, process, download results
- **Dashboard UI** — drag-and-drop interface with live segment viewer
- **Fault-tolerant stages** — each stage fails independently; pipeline continues
- **Structured outputs** — JSON, RTTM, CSV ready for ML training
- **CLI** — `python cli.py process audio.wav`
- **Docker** — one-command deployment

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/lingo.git
cd lingo

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set WHISPER_MODEL, PYANNOTE_TOKEN, etc.
```

> **Diarization**: Get a free HuggingFace token at https://huggingface.co/settings/tokens  
> Accept model terms at https://hf.co/pyannote/speaker-diarization-3.1  
> Then set `PYANNOTE_TOKEN=your_token` in `.env`

### 3. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
open http://localhost:8000
```

Or via CLI:

```bash
python cli.py process audio.wav --language ta
python cli.py batch ./audio_dir/ --language ta
python cli.py evaluate audio.wav "இது எதிர்பார்க்கப்பட்ட வரிகள்"
python cli.py serve
```

---

## Docker

```bash
docker-compose up --build
```

| Service | URL | Description |
|---------|-----|-------------|
| API + Dashboard | http://localhost:8000 | FastAPI + UI |
| Flower | http://localhost:5555 | Celery worker monitor |
| Redis | localhost:6379 | Job queue |

Scale workers:
```bash
docker-compose up --scale worker=4
```

---

## API Reference

### POST `/api/pipeline` — Full Pipeline

```bash
curl -X POST http://localhost:8000/api/pipeline \
  -F "file=@audio.wav" \
  -F "language=ta" \
  -F "run_diarization=true" \
  -F "run_emotion=true"
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "success",
  "transcription": {
    "text": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "language": "ta",
    "rtf": 0.23,
    "segments": [
      {
        "id": 0,
        "text": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "start": 0.0,
        "end": 2.4,
        "speaker": "SPEAKER_00",
        "emotion": "happy",
        "words": [...]
      }
    ]
  },
  "diarization": {
    "num_speakers": 2,
    "speaker_durations": { "SPEAKER_00": 12.4, "SPEAKER_01": 8.1 }
  },
  "emotion": {
    "dominant_emotion": "neutral",
    "emotion_distribution": { "neutral": 0.6, "happy": 0.3, "sad": 0.1 }
  },
  "timing": { "audio": 0.4, "transcription": 3.2, "diarization": 1.1, "emotion": 0.8 }
}
```

| Endpoint | Description |
|----------|-------------|
| `POST /api/pipeline` | Full pipeline |
| `POST /api/transcribe` | Transcription only |
| `POST /api/diarize` | Diarization only |
| `POST /api/emotion` | Emotion tagging only |
| `GET /api/models` | Model status |
| `GET /api/languages` | Supported languages |
| `GET /api/outputs/{job_id}/{file}` | Download output |

Full interactive docs: http://localhost:8000/docs

---

## Output Files

Each job writes to `data/outputs/{filename}/`:

| File | Description |
|------|-------------|
| `processed.wav` | Denoised 16kHz mono WAV |
| `transcript.json` | Full transcription with timestamps |
| `diarization.rttm` | Speaker turns (standard RTTM format) |
| `emotion_annotations.json` | Per-segment emotion + style |
| `emotion_annotations.csv` | Same as CSV for ML training |
| `full_output.json` | All stages merged |

---

## Project Structure

```
lingo/
├── app/
│   ├── config.py              # Settings (env-driven)
│   ├── main.py                # FastAPI app + endpoints
│   ├── tasks.py               # Celery async tasks
│   └── core/
│       ├── audio_processor.py # Ingestion, denoising, chunking
│       ├── transcriber.py     # Whisper ASR + WER evaluation
│       ├── diarizer.py        # pyannote diarization + RTTM
│       ├── emotion_tagger.py  # Emotion + style classification
│       └── pipeline.py        # Orchestrator
├── frontend/
│   └── index.html             # Dashboard UI
├── tests/
│   └── test_pipeline.py       # 20+ unit + integration tests
├── docker/
│   └── Dockerfile
├── cli.py                     # CLI (typer)
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | tiny/base/small/medium/large/large-v3 |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_LANGUAGE` | `ta` | Default language (None = auto-detect) |
| `PYANNOTE_TOKEN` | — | HuggingFace token for diarization |
| `API_PORT` | `8000` | API server port |
| `MAX_UPLOAD_SIZE_MB` | `500` | Max upload size |

**Model size guide:**

| Model | RAM | Speed | Quality |
|-------|-----|-------|---------|
| tiny | 1 GB | 32x | ★★☆☆☆ |
| base | 1 GB | 16x | ★★★☆☆ |
| small | 2 GB | 6x | ★★★★☆ |
| medium | 5 GB | 2x | ★★★★☆ |
| large-v3 | 10 GB | 1x | ★★★★★ |

---

## Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=app --cov-report=html
```

---

## Design Decisions

**Fault tolerance per stage** — if diarization fails on a bad file, transcription output is still valid. Critical for running unattended on thousands of files.

**RTTM output** — standard format consumed by most speech ML toolchains.

**Emotion → Style mapping** — anger + high energy = "aggressive"; happy + low energy = "cheerful". Style labels are directly useful for training expressive TTS models.

**Chunked processing** — audio > 30s is split, processed independently, timestamps stitched. Prevents OOM on long files.

**Idempotent job IDs** — each job writes to `outputs/{job_id}/`. History is preserved across runs.

---

## Roadmap

- [ ] YouTube/URL ingestion via `yt-dlp`
- [ ] Real-time streaming via WebSocket
- [ ] Speaker embedding extraction
- [ ] Code-switching detection (Tamil-English mixing)
- [ ] NeMo diarization alternative

---

## License

MIT
