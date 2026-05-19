"""
emotion_tagger.py — Speech emotion & style classification
Uses wav2vec2-based models for segment-level emotion tagging
Outputs: JSON annotations for TTS training data
"""
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from loguru import logger

from app.config import settings


STYLE_RULES = {
    # (emotion, energy_level) → speaking style
    ("angry", "high"):    "aggressive",
    ("angry", "low"):     "firm",
    ("happy", "high"):    "excited",
    ("happy", "low"):     "cheerful",
    ("sad", "high"):      "distressed",
    ("sad", "low"):       "melancholic",
    ("neutral", "high"):  "assertive",
    ("neutral", "low"):   "calm",
    ("fear", "high"):     "panicked",
    ("fear", "low"):      "anxious",
    ("surprise", "high"): "shocked",
    ("surprise", "low"):  "curious",
    ("disgust", "any"):   "disdainful",
}


@dataclass
class EmotionTag:
    segment_id: int
    start: float
    end: float
    speaker: Optional[str]
    emotion: str
    emotion_confidence: float
    emotion_scores: dict[str, float]    # all class probabilities
    speaking_style: str
    energy_level: str                   # "high" | "low"
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    speech_rate: Optional[float] = None # words per minute


@dataclass
class EmotionResult:
    tags: list[EmotionTag]
    dominant_emotion: str
    emotion_distribution: dict[str, float]
    mean_confidence: float
    processing_time_secs: float


class EmotionTagger:
    """
    Emotion classification using HuggingFace wav2vec2 model.
    Falls back to prosody-based heuristics if model unavailable.
    """

    def __init__(self):
        self._model = None
        self._feature_extractor = None
        self._available = False
        self.labels = settings.EMOTION_LABELS
        self._try_load()

    def _try_load(self):
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            logger.info(f"Loading emotion model: {settings.EMOTION_MODEL}")
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(settings.EMOTION_MODEL)
            self._model = AutoModelForAudioClassification.from_pretrained(settings.EMOTION_MODEL)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            self._available = True
            self.labels = list(self._model.config.id2label.values())
            logger.success(f"✓ Emotion model loaded | labels: {self.labels}")
        except Exception as e:
            logger.warning(f"Emotion model load failed ({e}) — using prosody-based fallback")

    def tag_segments(
        self,
        segments,           # list[TranscriptionSegment]
        waveform: np.ndarray,
        sample_rate: int,
    ) -> EmotionResult:
        start_t = time.time()
        tags = []

        for seg in segments:
            chunk = self._extract_chunk(waveform, sample_rate, seg.start, seg.end)
            if len(chunk) < sample_rate * 0.3:  # skip very short segments
                continue

            if self._available:
                emotion, confidence, scores = self._classify(chunk, sample_rate)
            else:
                emotion, confidence, scores = self._classify_prosody(chunk, sample_rate)

            energy = self._energy_level(chunk)
            style = self._infer_style(emotion, energy)
            pitch_mean, pitch_std = self._extract_pitch(chunk, sample_rate)
            speech_rate = self._speech_rate(seg)

            tag = EmotionTag(
                segment_id=seg.id,
                start=seg.start,
                end=seg.end,
                speaker=getattr(seg, "speaker", None),
                emotion=emotion,
                emotion_confidence=confidence,
                emotion_scores=scores,
                speaking_style=style,
                energy_level=energy,
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
                speech_rate=speech_rate,
            )
            tags.append(tag)
            seg.emotion = emotion

        emotion_dist = self._aggregate_distribution(tags)
        dominant = max(emotion_dist, key=emotion_dist.get) if emotion_dist else "neutral"
        mean_conf = float(np.mean([t.emotion_confidence for t in tags])) if tags else 0.0

        logger.success(
            f"✓ Emotion tagging | {len(tags)} segments | "
            f"dominant={dominant} | conf={mean_conf:.2f}"
        )
        return EmotionResult(
            tags=tags,
            dominant_emotion=dominant,
            emotion_distribution=emotion_dist,
            mean_confidence=mean_conf,
            processing_time_secs=time.time() - start_t,
        )

    def _classify(self, chunk: np.ndarray, sample_rate: int) -> tuple[str, float, dict]:
        import torch
        inputs = self._feature_extractor(
            chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        idx = int(np.argmax(probs))
        scores = {label: float(probs[i]) for i, label in enumerate(self.labels)}
        return self.labels[idx], float(probs[idx]), scores

    def _classify_prosody(self, chunk: np.ndarray, sample_rate: int) -> tuple[str, float, dict]:
        """Heuristic emotion from pitch/energy when model unavailable."""
        try:
            import librosa
            pitch_mean, pitch_std = self._extract_pitch(chunk, sample_rate)
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            # Simple heuristic rules
            if rms > 0.15 and pitch_std and pitch_std > 50:
                emotion = "angry"
            elif pitch_mean and pitch_mean > 200 and rms > 0.1:
                emotion = "happy"
            elif rms < 0.03:
                emotion = "sad"
            else:
                emotion = "neutral"

            scores = {e: 0.05 for e in self.labels}
            scores[emotion] = 0.7
            return emotion, 0.7, scores
        except Exception:
            scores = {e: 1/len(self.labels) for e in self.labels}
            return "neutral", 1/len(self.labels), scores

    def _extract_chunk(self, waveform, sr, start, end) -> np.ndarray:
        s = max(0, int(start * sr))
        e = min(len(waveform), int(end * sr))
        return waveform[s:e]

    def _energy_level(self, chunk: np.ndarray) -> str:
        rms = np.sqrt(np.mean(chunk ** 2))
        return "high" if rms > 0.08 else "low"

    def _extract_pitch(self, chunk: np.ndarray, sr: int) -> tuple[Optional[float], Optional[float]]:
        try:
            import librosa
            f0, voiced, _ = librosa.pyin(chunk, fmin=50, fmax=500, sr=sr)
            voiced_f0 = f0[voiced > 0.5] if voiced is not None else f0[~np.isnan(f0)]
            if len(voiced_f0) > 0:
                return float(np.mean(voiced_f0)), float(np.std(voiced_f0))
        except Exception:
            pass
        return None, None

    def _speech_rate(self, seg) -> Optional[float]:
        dur = seg.end - seg.start
        if dur <= 0:
            return None
        words = len(seg.text.split())
        return round(words / dur * 60, 1)  # WPM

    def _infer_style(self, emotion: str, energy: str) -> str:
        style = STYLE_RULES.get((emotion, energy))
        if not style:
            style = STYLE_RULES.get((emotion, "any"), "neutral")
        return style or "neutral"

    def _aggregate_distribution(self, tags: list[EmotionTag]) -> dict[str, float]:
        if not tags:
            return {}
        counts = {}
        for tag in tags:
            counts[tag.emotion] = counts.get(tag.emotion, 0) + 1
        total = len(tags)
        return {k: round(v / total, 3) for k, v in counts.items()}

    def export_annotations(
        self,
        result: EmotionResult,
        output_path: Path,
        format: str = "json",
    ) -> Path:
        """Export emotion annotations as JSON or CSV for TTS training."""
        import json, csv
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = {
                "summary": {
                    "dominant_emotion": result.dominant_emotion,
                    "emotion_distribution": result.emotion_distribution,
                    "mean_confidence": result.mean_confidence,
                },
                "segments": [
                    {
                        "segment_id": t.segment_id,
                        "start": t.start,
                        "end": t.end,
                        "speaker": t.speaker,
                        "emotion": t.emotion,
                        "confidence": t.emotion_confidence,
                        "scores": t.emotion_scores,
                        "style": t.speaking_style,
                        "energy": t.energy_level,
                        "pitch_mean": t.pitch_mean,
                        "pitch_std": t.pitch_std,
                        "speech_rate_wpm": t.speech_rate,
                    }
                    for t in result.tags
                ],
            }
            output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        elif format == "csv":
            with open(output_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "segment_id", "start", "end", "speaker",
                    "emotion", "confidence", "style", "energy",
                    "pitch_mean", "pitch_std", "speech_rate_wpm"
                ])
                w.writeheader()
                for t in result.tags:
                    w.writerow({
                        "segment_id": t.segment_id,
                        "start": t.start, "end": t.end,
                        "speaker": t.speaker,
                        "emotion": t.emotion,
                        "confidence": round(t.emotion_confidence, 4),
                        "style": t.speaking_style,
                        "energy": t.energy_level,
                        "pitch_mean": t.pitch_mean,
                        "pitch_std": t.pitch_std,
                        "speech_rate_wpm": t.speech_rate,
                    })

        logger.info(f"Annotations saved → {output_path}")
        return output_path
