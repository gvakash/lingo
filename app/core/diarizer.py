"""
diarizer.py
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
from loguru import logger
from app.config import settings

@dataclass
class DiarizationTurn:
    speaker: str
    start: float
    end: float
    duration: float

@dataclass
class DiarizationResult:
    turns: list
    num_speakers: int
    speaker_durations: dict
    rttm_content: str

class SpeakerDiarizer:

    def __init__(self):
        self._pipeline = None
        self._available = False
        self._try_load()

    def _try_load(self):
        if not settings.PYANNOTE_TOKEN:
            logger.warning("PYANNOTE_TOKEN not set — using fallback diarization.")
            return
        try:
            from pyannote.audio import Pipeline
            import torch
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=settings.PYANNOTE_TOKEN)
            if torch.cuda.is_available():
                self._pipeline = self._pipeline.to(torch.device("cuda"))
            self._available = True
            logger.success("pyannote diarization pipeline loaded")
        except Exception as e:
            logger.warning(f"pyannote load failed ({e}) — using fallback")

    def diarize(self, waveform, sample_rate, num_speakers=None):
        if self._available:
            return self._diarize_pyannote(waveform, sample_rate, num_speakers)
        return self._diarize_fallback(waveform, sample_rate)

    def _diarize_pyannote(self, waveform, sample_rate, num_speakers):
        import torch
        waveform_tensor = torch.tensor(waveform[np.newaxis, :], dtype=torch.float32)
        audio_in_memory = {"waveform": waveform_tensor, "sample_rate": sample_rate}
        kwargs = {"num_speakers": num_speakers} if num_speakers else {
            "min_speakers": settings.MIN_SPEAKERS, "max_speakers": settings.MAX_SPEAKERS}
        annotation = self._pipeline(audio_in_memory, **kwargs)
        turns = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(DiarizationTurn(speaker=speaker,
                start=round(segment.start, 3), end=round(segment.end, 3),
                duration=round(segment.duration, 3)))
        turns.sort(key=lambda t: t.start)
        speakers = sorted({t.speaker for t in turns})
        speaker_durations = {s: round(sum(t.duration for t in turns if t.speaker == s), 3) for s in speakers}
        return DiarizationResult(turns=turns, num_speakers=len(speakers),
            speaker_durations=speaker_durations, rttm_content=self._to_rttm(turns, "audio"))

    def _diarize_fallback(self, waveform, sample_rate):
        logger.warning("Using energy-based fallback diarization — single speaker assumed")
        chunk_size = sample_rate * 2
        turns = []
        for i in range(0, len(waveform), chunk_size):
            chunk = waveform[i:i+chunk_size]
            if np.sqrt(np.mean(chunk**2)) > 0.01:
                start = i / sample_rate
                end = min(i + chunk_size, len(waveform)) / sample_rate
                turns.append(DiarizationTurn(speaker="SPEAKER_00",
                    start=round(start, 3), end=round(end, 3), duration=round(end-start, 3)))
        speaker_durations = {"SPEAKER_00": round(sum(t.duration for t in turns), 3)}
        return DiarizationResult(turns=turns, num_speakers=1,
            speaker_durations=speaker_durations, rttm_content=self._to_rttm(turns, "audio"))

    def assign_speakers(self, segments, diarization):
        for seg in segments:
            best_speaker, best_overlap = None, 0.0
            for turn in diarization.turns:
                overlap = self._overlap(seg.start, seg.end, turn.start, turn.end)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn.speaker
            seg.speaker = best_speaker or "UNKNOWN"
        return segments

    def save_rttm(self, result, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.rttm_content)
        return output_path

    @staticmethod
    def _overlap(a_start, a_end, b_start, b_end):
        return max(0, min(a_end, b_end) - max(a_start, b_start))

    @staticmethod
    def _to_rttm(turns, file_id):
        return "\n".join(
            f"SPEAKER {file_id} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
            for t in turns)
