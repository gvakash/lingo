import sys
import unittest.mock
if "numba" not in sys.modules:
    sys.modules["numba"] = unittest.mock.MagicMock()
    sys.modules["numba.core"] = unittest.mock.MagicMock()
    sys.modules["numba.core.types"] = unittest.mock.MagicMock()

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from loguru import logger
from app.config import settings


@dataclass
class WordToken:
    word: str
    start: float
    end: float
    confidence: float


@dataclass


class TranscriptionSegment:
    id: int
    text: str
    start: float
    end: float
    language: str
    avg_logprob: float
    no_speech_prob: float
    words: list = field(default_factory=list)
    speaker: Optional[str] = None
    emotion: Optional[str] = None


@dataclass
class TranscriptionResult:
    full_text: str
    language: str
    language_confidence: float
    segments: list
    duration_secs: float
    processing_time_secs: float
    model_name: str
    rtf: float
    wer: Optional[float] = None
    cer: Optional[float] = None


class WhisperTranscriber:

    def __init__(self):
        self.model_name = settings.WHISPER_MODEL
        self.device = settings.WHISPER_DEVICE
        self._model = None
        logger.info(f"WhisperTranscriber init | model={self.model_name} | device={self.device}")

    def _load_model(self):
        if self._model is None:
            import whisper
            logger.info(f"Loading Whisper {self.model_name}...")
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.success(f"Whisper {self.model_name} loaded on {self.device}")
        return self._model

    def transcribe(self, waveform, sample_rate, language=None, reference_text=None, word_timestamps=True):
        import whisper
        model = self._load_model()
        start_t = time.time()
        duration = len(waveform) / sample_rate
        use_lang = language or settings.WHISPER_LANGUAGE
        logger.info(f"Transcribing {duration:.1f}s audio | lang={use_lang or 'auto'}")

        audio = waveform.astype(np.float32)
        audio_padded = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_padded, n_mels=model.dims.n_mels).to(model.device)

        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        final_lang = use_lang or detected_lang

        options = whisper.DecodingOptions(language=final_lang, fp16=False)
        decode_result = whisper.decode(model, mel, options)

        processing_time = time.time() - start_t
        rtf = processing_time / max(duration, 0.01)

        seg = TranscriptionSegment(
            id=0, text=decode_result.text.strip(),
            start=0.0, end=duration, language=detected_lang,
            avg_logprob=decode_result.avg_logprob,
            no_speech_prob=decode_result.no_speech_prob,
        )

        tx = TranscriptionResult(
            full_text=decode_result.text.strip(),
            language=detected_lang,
            language_confidence=float(probs.get(detected_lang, 0)),
            segments=[seg],
            duration_secs=duration,
            processing_time_secs=processing_time,
            model_name=self.model_name,
            rtf=rtf,
        )

        if reference_text:
            tx.wer, tx.cer = self._compute_metrics(tx.full_text, reference_text)

        logger.success(f"Done | lang={detected_lang} | RTF={rtf:.2f}x | {tx.full_text[:80]}")
        return tx

    def transcribe_chunks(self, chunks, sample_rate, language=None):
        results = []
        offset = 0.0
        for i, chunk in enumerate(chunks):
            r = self.transcribe(chunk, sample_rate, language=language)
            for seg in r.segments:
                seg.start += offset
                seg.end += offset
            results.append(r)
            offset += len(chunk) / sample_rate
        all_segs = [seg for r in results for seg in r.segments]
        total_dur = sum(len(c)/sample_rate for c in chunks)
        return TranscriptionResult(
            full_text=" ".join(r.full_text for r in results),
            language=results[0].language if results else "unknown",
            language_confidence=results[0].language_confidence if results else 0.0,
            segments=all_segs,
            duration_secs=total_dur,
            processing_time_secs=sum(r.processing_time_secs for r in results),
            model_name=self.model_name,
            rtf=sum(r.processing_time_secs for r in results) / max(total_dur, 0.01),
        )

    def transcribe_file(self, audio_path, **kwargs):
        import soundfile as sf
        waveform, sr = sf.read(str(audio_path), dtype="float32")
        return self.transcribe(waveform, sr, **kwargs)

    def _compute_metrics(self, hypothesis, reference):
        try:
            from jiwer import wer, cer
            return float(wer(reference, hypothesis)), float(cer(reference, hypothesis))
        except Exception:
            return 0.0, 0.0
