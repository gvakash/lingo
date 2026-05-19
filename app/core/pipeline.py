"""
pipeline.py — Master orchestrator for the Lingo Speech Pipeline
Chains: Audio Processing → Transcription → Diarization → Emotion Tagging → Export
"""
import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Any

from loguru import logger

from app.config import settings
from app.core.audio_processor import AudioProcessor, ProcessedAudio
from app.core.transcriber import WhisperTranscriber, TranscriptionResult
from app.core.diarizer import SpeakerDiarizer, DiarizationResult
from app.core.emotion_tagger import EmotionTagger, EmotionResult


@dataclass
class PipelineConfig:
    run_diarization: bool = True
    run_emotion: bool = True
    denoise: bool = True
    normalize: bool = True
    word_timestamps: bool = True
    language: Optional[str] = None
    num_speakers: Optional[int] = None
    reference_text: Optional[str] = None
    export_formats: list[str] = field(default_factory=lambda: ["json", "rttm", "csv"])


@dataclass
class PipelineOutput:
    job_id: str
    status: str
    input_file: str
    config: dict

    audio_metadata: Optional[dict] = None
    transcription: Optional[dict] = None
    diarization: Optional[dict] = None
    emotion: Optional[dict] = None

    output_files: dict[str, str] = field(default_factory=dict)

    total_time_secs: float = 0.0
    stage_times: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class SpeechPipeline:
    """
    End-to-end speech processing pipeline.
    Each stage is independently fault-tolerant.
    """

    def __init__(self):
        logger.info("Initializing Lingo SpeechPipeline...")
        self.audio_proc = AudioProcessor()
        self.transcriber = WhisperTranscriber()
        self.diarizer = SpeakerDiarizer()
        self.emotion_tagger = EmotionTagger()
        logger.success("✓ SpeechPipeline ready")

    def run(
        self,
        input_path: Path | str,
        config: PipelineConfig = None,
        output_dir: Path | str = None,
    ) -> PipelineOutput:
        config = config or PipelineConfig()
        input_path = Path(input_path)
        output_dir = Path(output_dir or settings.OUTPUT_DIR) / input_path.stem

        job_id = str(uuid.uuid4())[:8]
        output = PipelineOutput(
            job_id=job_id,
            status="running",
            input_file=str(input_path),
            config=asdict(config),
        )

        logger.info(f"{'='*60}")
        logger.info(f"JOB {job_id} | {input_path.name}")
        logger.info(f"{'='*60}")
        total_start = time.time()

        # ── Stage 1: Audio Processing ──────────────────────────────────────
        t = time.time()
        try:
            processed = self.audio_proc.process_file(
                input_path, denoise=config.denoise, normalize=config.normalize
            )
            output.audio_metadata = self._audio_meta_dict(processed)
            output.stage_times["audio"] = round(time.time() - t, 2)

            processed_path = output_dir / "processed.wav"
            self.audio_proc.save_processed(processed, processed_path)
            output.output_files["processed_wav"] = str(processed_path)

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            output.errors["audio"] = str(e)
            output.status = "failed"
            return self._finalize(output, total_start)

        # ── Stage 2: Transcription ─────────────────────────────────────────
        t = time.time()
        transcription_result: Optional[TranscriptionResult] = None
        try:
            if processed.chunks:
                transcription_result = self.transcriber.transcribe_chunks(
                    processed.chunks, processed.sample_rate,
                    language=config.language,
                )
            else:
                transcription_result = self.transcriber.transcribe(
                    processed.waveform, processed.sample_rate,
                    language=config.language,
                    reference_text=config.reference_text,
                    word_timestamps=config.word_timestamps,
                )
            output.transcription = self._transcription_dict(transcription_result)
            output.stage_times["transcription"] = round(time.time() - t, 2)

            transcript_path = output_dir / "transcript.json"
            self._save_json(output.transcription, transcript_path)
            output.output_files["transcript"] = str(transcript_path)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            output.errors["transcription"] = str(e)

        # ── Stage 3: Diarization ───────────────────────────────────────────
        t = time.time()
        diarization_result: Optional[DiarizationResult] = None
        if config.run_diarization:
            try:
                diarization_result = self.diarizer.diarize(
                    processed.waveform, processed.sample_rate,
                    num_speakers=config.num_speakers,
                )
                output.diarization = self._diarization_dict(diarization_result)
                output.stage_times["diarization"] = round(time.time() - t, 2)

                if transcription_result:
                    self.diarizer.assign_speakers(
                        transcription_result.segments, diarization_result
                    )

                if "rttm" in config.export_formats:
                    rttm_path = output_dir / "diarization.rttm"
                    self.diarizer.save_rttm(diarization_result, rttm_path)
                    output.output_files["rttm"] = str(rttm_path)

            except Exception as e:
                logger.error(f"Diarization failed: {e}")
                output.errors["diarization"] = str(e)

        # ── Stage 4: Emotion Tagging ───────────────────────────────────────
        t = time.time()
        if config.run_emotion and transcription_result:
            try:
                emotion_result = self.emotion_tagger.tag_segments(
                    transcription_result.segments,
                    processed.waveform,
                    processed.sample_rate,
                )
                output.emotion = self._emotion_dict(emotion_result)
                output.stage_times["emotion"] = round(time.time() - t, 2)

                if "json" in config.export_formats:
                    ann_path = output_dir / "emotion_annotations.json"
                    self.emotion_tagger.export_annotations(emotion_result, ann_path, "json")
                    output.output_files["emotion_json"] = str(ann_path)

                if "csv" in config.export_formats:
                    csv_path = output_dir / "emotion_annotations.csv"
                    self.emotion_tagger.export_annotations(emotion_result, csv_path, "csv")
                    output.output_files["emotion_csv"] = str(csv_path)

            except Exception as e:
                logger.error(f"Emotion tagging failed: {e}")
                output.errors["emotion"] = str(e)

        # ── Save merged output ─────────────────────────────────────────────
        if transcription_result:
            merged = self._build_merged_output(transcription_result, output)
            merged_path = output_dir / "full_output.json"
            self._save_json(merged, merged_path)
            output.output_files["full_output"] = str(merged_path)

        output.status = "success" if not output.errors else "partial"
        return self._finalize(output, total_start)

    def run_from_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        config: PipelineConfig = None,
    ) -> PipelineOutput:
        tmp_path = settings.RAW_DIR / filename
        tmp_path.write_bytes(audio_bytes)
        return self.run(tmp_path, config)

    def _audio_meta_dict(self, processed: ProcessedAudio) -> dict:
        m = processed.metadata
        return {
            "file": m.file_path,
            "hash": m.file_hash,
            "duration_secs": m.duration_secs,
            "sample_rate": m.sample_rate,
            "channels": m.channels,
            "format": m.format,
            "size_bytes": m.file_size_bytes,
            "snr_db": m.snr_db,
            "loudness_dbfs": m.loudness_dbfs,
            "num_chunks": len(processed.chunks),
        }

    def _transcription_dict(self, tx: TranscriptionResult) -> dict:
        return {
            "text": tx.full_text,
            "language": tx.language,
            "language_confidence": tx.language_confidence,
            "duration_secs": tx.duration_secs,
            "rtf": round(tx.rtf, 3),
            "wer": tx.wer,
            "cer": tx.cer,
            "model": tx.model_name,
            "segments": [
                {
                    "id": s.id,
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "avg_logprob": round(s.avg_logprob, 4),
                    "no_speech_prob": round(s.no_speech_prob, 4),
                    "speaker": s.speaker,
                    "emotion": s.emotion,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end,
                         "confidence": round(w.confidence, 4)}
                        for w in s.words
                    ],
                }
                for s in tx.segments
            ],
        }

    def _diarization_dict(self, d: DiarizationResult) -> dict:
        return {
            "num_speakers": d.num_speakers,
            "speaker_durations": d.speaker_durations,
            "turns": [
                {"speaker": t.speaker, "start": t.start, "end": t.end, "duration": t.duration}
                for t in d.turns
            ],
        }

    def _emotion_dict(self, e: EmotionResult) -> dict:
        return {
            "dominant_emotion": e.dominant_emotion,
            "emotion_distribution": e.emotion_distribution,
            "mean_confidence": round(e.mean_confidence, 4),
        }

    def _build_merged_output(self, tx: TranscriptionResult, output: PipelineOutput) -> dict:
        return {
            "job_id": output.job_id,
            "input_file": output.input_file,
            "audio": output.audio_metadata,
            "transcription": {
                "text": tx.full_text,
                "language": tx.language,
                "wer": tx.wer,
                "rtf": tx.rtf,
            },
            "diarization": output.diarization,
            "emotion": output.emotion,
            "segments": output.transcription.get("segments", []) if output.transcription else [],
            "output_files": output.output_files,
            "timing": output.stage_times,
        }

    def _finalize(self, output: PipelineOutput, total_start: float) -> PipelineOutput:
        output.total_time_secs = round(time.time() - total_start, 2)
        status_icon = "✓" if output.status == "success" else "⚠" if output.status == "partial" else "✗"
        logger.info(
            f"{status_icon} Job {output.job_id} {output.status.upper()} | "
            f"{output.total_time_secs}s | errors: {list(output.errors.keys()) or 'none'}"
        )
        return output

    @staticmethod
    def _save_json(data: dict | list, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
