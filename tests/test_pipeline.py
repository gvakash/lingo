"""
tests/test_pipeline.py — Integration tests for the speech pipeline
Run: pytest tests/ -v
"""
import io
import json
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sine_wav_bytes():
    """Generate a 2-second 440Hz sine wave as WAV bytes."""
    sr = 16000
    t = np.linspace(0, 2, sr * 2)
    wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def sample_waveform():
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr


@pytest.fixture
def tmp_wav(tmp_path, sine_wav_bytes):
    p = tmp_path / "test.wav"
    p.write_bytes(sine_wav_bytes)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# AudioProcessor Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioProcessor:
    def test_process_wav_file(self, tmp_wav):
        from app.core.audio_processor import AudioProcessor
        proc = AudioProcessor()
        result = proc.process_file(tmp_wav, denoise=False)
        assert result.waveform is not None
        assert result.sample_rate == 16000
        assert result.metadata.is_valid
        assert result.metadata.duration_secs > 0

    def test_mono_conversion(self):
        from app.core.audio_processor import AudioProcessor
        proc = AudioProcessor()
        stereo = np.random.randn(2, 16000).astype(np.float32)
        mono = proc._to_mono(stereo)
        assert mono.ndim == 1
        assert len(mono) == 16000

    def test_normalization(self):
        from app.core.audio_processor import AudioProcessor
        proc = AudioProcessor()
        loud = (np.random.randn(16000) * 5).astype(np.float32)
        normalized = proc._normalize(loud)
        assert np.max(np.abs(normalized)) <= 1.0

    def test_resample(self):
        from app.core.audio_processor import AudioProcessor
        proc = AudioProcessor()
        wave_44k = np.random.randn(44100).astype(np.float32)
        resampled = proc._resample(wave_44k, 44100)
        assert len(resampled) == 16000

    def test_chunking(self):
        from app.core.audio_processor import AudioProcessor
        proc = AudioProcessor()
        # 65 second audio should produce 3 chunks (30s + 30s + 5s)
        wave = np.zeros(16000 * 65, dtype=np.float32)
        chunks = proc._chunk(wave)
        assert len(chunks) == 3

    def test_unsupported_format(self, tmp_path):
        from app.core.audio_processor import AudioProcessor
        proc = AudioProcessor()
        bad_file = tmp_path / "test.xyz"
        bad_file.write_bytes(b"fake audio")
        meta = proc._extract_metadata(bad_file)
        assert not meta.is_valid
        assert "Unsupported" in meta.error

    def test_hash_consistency(self, tmp_wav):
        from app.core.audio_processor import AudioProcessor
        h1 = AudioProcessor._hash_file(tmp_wav)
        h2 = AudioProcessor._hash_file(tmp_wav)
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex


# ──────────────────────────────────────────────────────────────────────────────
# Transcriber Tests (mocked Whisper)
# ──────────────────────────────────────────────────────────────────────────────

class TestTranscriber:
    def _mock_whisper_result(self):
        return {
            "text": "Hello world this is a test",
            "language": "en",
            "segments": [
                {
                    "id": 0,
                    "text": "Hello world this is a test",
                    "start": 0.0,
                    "end": 2.0,
                    "avg_logprob": -0.3,
                    "no_speech_prob": 0.05,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                        {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.90},
                    ]
                }
            ]
        }

    def test_transcription_structure(self, sample_waveform):
        from app.core.transcriber import WhisperTranscriber
        tx = WhisperTranscriber()

        with patch.object(tx, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = self._mock_whisper_result()
            mock_load.return_value = mock_model
            tx._model = mock_model

            waveform, sr = sample_waveform
            result = tx.transcribe(waveform, sr)

        assert result.full_text == "Hello world this is a test"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.duration_secs > 0
        assert result.rtf > 0

    def test_wer_computation(self, sample_waveform):
        from app.core.transcriber import WhisperTranscriber
        tx = WhisperTranscriber()

        with patch.object(tx, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = self._mock_whisper_result()
            mock_load.return_value = mock_model
            tx._model = mock_model

            waveform, sr = sample_waveform
            result = tx.transcribe(
                waveform, sr,
                reference_text="Hello world this is a test"
            )

        # Perfect match → WER should be 0
        assert result.wer is not None
        assert result.wer == pytest.approx(0.0, abs=0.01)

    def test_chunk_stitching(self, sample_waveform):
        from app.core.transcriber import WhisperTranscriber
        tx = WhisperTranscriber()
        waveform, sr = sample_waveform
        chunks = [waveform[:sr], waveform[sr:2*sr], waveform[2*sr:]]

        with patch.object(tx, 'transcribe') as mock_tx:
            from app.core.transcriber import TranscriptionResult, TranscriptionSegment
            mock_tx.return_value = TranscriptionResult(
                full_text="chunk text",
                language="en",
                language_confidence=1.0,
                segments=[TranscriptionSegment(
                    id=0, text="chunk text", start=0.0, end=1.0,
                    language="en", avg_logprob=-0.3, no_speech_prob=0.05
                )],
                duration_secs=1.0,
                processing_time_secs=0.1,
                model_name="base",
                rtf=0.1,
            )
            result = tx.transcribe_chunks(chunks, sr)

        assert result.full_text == "chunk text chunk text chunk text"
        assert len(result.segments) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Diarizer Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDiarizer:
    def test_fallback_diarization(self, sample_waveform):
        from app.core.diarizer import SpeakerDiarizer
        d = SpeakerDiarizer()
        d._available = False  # force fallback
        waveform, sr = sample_waveform
        result = d.diarize(waveform, sr)
        assert result.num_speakers >= 1
        assert isinstance(result.rttm_content, str)
        assert len(result.turns) >= 0

    def test_rttm_format(self, sample_waveform):
        from app.core.diarizer import SpeakerDiarizer
        d = SpeakerDiarizer()
        d._available = False
        waveform, sr = sample_waveform
        result = d.diarize(waveform, sr)
        if result.rttm_content:
            line = result.rttm_content.split("\n")[0]
            parts = line.split()
            assert parts[0] == "SPEAKER"
            assert len(parts) == 10

    def test_speaker_assignment(self, sample_waveform):
        from app.core.diarizer import SpeakerDiarizer, DiarizationResult, DiarizationTurn
        from app.core.transcriber import TranscriptionSegment

        d = SpeakerDiarizer()
        seg = TranscriptionSegment(
            id=0, text="hello", start=0.0, end=1.0,
            language="en", avg_logprob=-0.3, no_speech_prob=0.05
        )
        diar = DiarizationResult(
            turns=[DiarizationTurn("SPEAKER_00", 0.0, 1.5, 1.5)],
            num_speakers=1,
            speaker_durations={"SPEAKER_00": 1.5},
            rttm_content=""
        )
        result = d.assign_speakers([seg], diar)
        assert result[0].speaker == "SPEAKER_00"

    def test_rttm_save(self, tmp_path, sample_waveform):
        from app.core.diarizer import SpeakerDiarizer
        d = SpeakerDiarizer()
        d._available = False
        waveform, sr = sample_waveform
        result = d.diarize(waveform, sr)
        rttm_path = tmp_path / "test.rttm"
        d.save_rttm(result, rttm_path)
        assert rttm_path.exists()


# ──────────────────────────────────────────────────────────────────────────────
# Emotion Tagger Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEmotionTagger:
    def test_prosody_fallback(self, sample_waveform):
        from app.core.emotion_tagger import EmotionTagger
        from app.core.transcriber import TranscriptionSegment

        tagger = EmotionTagger()
        tagger._available = False

        seg = TranscriptionSegment(
            id=0, text="hello world", start=0.0, end=2.0,
            language="en", avg_logprob=-0.3, no_speech_prob=0.05
        )
        waveform, sr = sample_waveform
        result = tagger.tag_segments([seg], waveform, sr)

        assert len(result.tags) == 1
        assert result.tags[0].emotion in tagger.labels
        assert 0.0 <= result.tags[0].emotion_confidence <= 1.0
        assert result.tags[0].speaking_style is not None

    def test_emotion_export_json(self, tmp_path, sample_waveform):
        from app.core.emotion_tagger import EmotionTagger
        from app.core.transcriber import TranscriptionSegment

        tagger = EmotionTagger()
        tagger._available = False

        seg = TranscriptionSegment(
            id=0, text="test utterance", start=0.0, end=2.0,
            language="en", avg_logprob=-0.3, no_speech_prob=0.05
        )
        waveform, sr = sample_waveform
        result = tagger.tag_segments([seg], waveform, sr)

        path = tmp_path / "emotions.json"
        tagger.export_annotations(result, path, "json")

        assert path.exists()
        data = json.loads(path.read_text())
        assert "summary" in data
        assert "segments" in data

    def test_style_inference(self):
        from app.core.emotion_tagger import EmotionTagger
        tagger = EmotionTagger()
        style = tagger._infer_style("angry", "high")
        assert style == "aggressive"
        style = tagger._infer_style("happy", "low")
        assert style == "cheerful"


# ──────────────────────────────────────────────────────────────────────────────
# API Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from app.main import app, pipeline
        from app.core.pipeline import SpeechPipeline, PipelineOutput, PipelineConfig
        import app.main as main_module

        # Mock the pipeline
        mock_pipeline = MagicMock(spec=SpeechPipeline)
        mock_pipeline.run_from_bytes.return_value = PipelineOutput(
            job_id="test123",
            status="success",
            input_file="test.wav",
            config={},
            audio_metadata={"duration_secs": 2.0, "sample_rate": 16000},
            transcription={"text": "hello world", "language": "en", "rtf": 0.1, "segments": []},
            diarization={"num_speakers": 1, "turns": [], "speaker_durations": {}},
            emotion={"dominant_emotion": "neutral", "emotion_distribution": {}, "mean_confidence": 0.8},
            total_time_secs=1.0,
        )
        main_module.pipeline = mock_pipeline
        return TestClient(app)

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_pipeline_endpoint(self, client, sine_wav_bytes):
        response = client.post(
            "/api/pipeline",
            files={"file": ("test.wav", sine_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "transcription" in data
        assert "diarization" in data
        assert "emotion" in data

    def test_transcribe_endpoint(self, client, sine_wav_bytes):
        response = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", sine_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200

    def test_invalid_format(self, client):
        response = client.post(
            "/api/pipeline",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert response.status_code == 400

    def test_models_endpoint(self, client):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "whisper" in data
        assert "diarization" in data
        assert "emotion" in data

    def test_languages_endpoint(self, client):
        response = client.get("/api/languages")
        assert response.status_code == 200
        data = response.json()
        assert "indian_languages" in data
        assert "hi" in data["indian_languages"]
