"""
audio_processor.py
"""
import hashlib, subprocess, json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from loguru import logger
from app.config import settings

@dataclass
class AudioMetadata:
    file_path: str
    file_hash: str
    duration_secs: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int
    is_valid: bool = True
    error: Optional[str] = None
    loudness_dbfs: Optional[float] = None
    snr_db: Optional[float] = None

@dataclass
class ProcessedAudio:
    waveform: np.ndarray
    sample_rate: int
    metadata: AudioMetadata
    chunks: list = field(default_factory=list)

class AudioProcessor:
    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm"}

    def __init__(self):
        self.target_sr = settings.TARGET_SAMPLE_RATE
        self.chunk_dur = settings.CHUNK_DURATION_SECS
        logger.info(f"AudioProcessor init | target_sr={self.target_sr}Hz | chunk={self.chunk_dur}s")

    def process_file(self, path, denoise=True, normalize=True):
        path = Path(path)
        logger.info(f"Processing audio: {path.name}")
        meta = self._extract_metadata(path)
        if not meta.is_valid:
            raise ValueError(f"Invalid audio file: {meta.error}")
        waveform, sr = self._load_audio(path)
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sr)
        if normalize:
            waveform = self._normalize(waveform)
        if denoise:
            waveform = self._denoise(waveform)
        meta.snr_db = self._estimate_snr(waveform)
        meta.loudness_dbfs = self._loudness(waveform)
        chunks = self._chunk(waveform) if meta.duration_secs > self.chunk_dur else []
        logger.success(f"Processed {path.name} | {meta.duration_secs:.1f}s")
        return ProcessedAudio(waveform=waveform, sample_rate=self.target_sr, metadata=meta, chunks=chunks)

    def process_bytes(self, audio_bytes, filename="upload.wav", denoise=True):
        tmp = Path("/tmp") / filename
        tmp.write_bytes(audio_bytes)
        try:
            return self.process_file(tmp, denoise=denoise)
        finally:
            tmp.unlink(missing_ok=True)

    def save_processed(self, audio, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio.waveform, audio.sample_rate, subtype="PCM_16")
        logger.info(f"Saved to {output_path}")
        return output_path

    def _extract_metadata(self, path):
        suffix = path.suffix.lower()
        file_hash = self._hash_file(path)
        file_size = path.stat().st_size
        if suffix not in self.SUPPORTED_FORMATS:
            return AudioMetadata(file_path=str(path), file_hash=file_hash, duration_secs=0,
                sample_rate=0, channels=0, format=suffix, file_size_bytes=file_size,
                is_valid=False, error=f"Unsupported format: {suffix}")
        try:
            info = sf.info(str(path))
            duration, sr, channels = info.duration, info.samplerate, info.channels
        except Exception:
            try:
                result = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_streams", str(path)], capture_output=True, text=True, timeout=30)
                data = json.loads(result.stdout)
                stream = data["streams"][0]
                duration = float(stream.get("duration", 0))
                sr = int(stream.get("sample_rate", 0))
                channels = int(stream.get("channels", 1))
            except Exception as e:
                return AudioMetadata(file_path=str(path), file_hash=file_hash, duration_secs=0,
                    sample_rate=0, channels=0, format=suffix, file_size_bytes=file_size,
                    is_valid=False, error=str(e))
        return AudioMetadata(file_path=str(path), file_hash=file_hash, duration_secs=duration,
            sample_rate=sr, channels=channels, format=suffix, file_size_bytes=file_size, is_valid=True)

    def _load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(str(path))
            return waveform.numpy(), sr
        except Exception as e:
            logger.warning(f"torchaudio failed ({e}), trying pydub...")
            return self._load_via_pydub(path)

    def _load_via_pydub(self, path):
        audio = AudioSegment.from_file(str(path))
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= 2 ** (audio.sample_width * 8 - 1)
        if audio.channels > 1:
            samples = samples.reshape(-1, audio.channels).T
        return samples, audio.frame_rate

    def _to_mono(self, waveform):
        if waveform.ndim == 1:
            return waveform
        return np.mean(waveform, axis=0)

    def _resample(self, waveform, orig_sr):
        if orig_sr == self.target_sr:
            return waveform
        tensor = torch.tensor(waveform).unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, orig_sr, self.target_sr)
        return resampled.squeeze().numpy().astype(np.float32)

    def _normalize(self, waveform):
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak * 0.707
        return waveform.astype(np.float32)

    def _denoise(self, waveform):
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=waveform, sr=self.target_sr, stationary=False, prop_decrease=0.75).astype(np.float32)
        except Exception as e:
            logger.warning(f"Noise reduction skipped: {e}")
            return waveform

    def _chunk(self, waveform):
        chunk_samples = self.chunk_dur * self.target_sr
        return [waveform[i:i+chunk_samples] for i in range(0, len(waveform), chunk_samples)]

    def _estimate_snr(self, waveform):
        rms = np.sqrt(np.mean(waveform**2))
        noise = np.sqrt(np.mean(np.sort(np.abs(waveform))[:len(waveform)//10]**2))
        return float(20 * np.log10(rms / (noise + 1e-10))) if noise >= 1e-10 else 60.0

    def _loudness(self, waveform):
        return float(20 * np.log10(np.sqrt(np.mean(waveform**2)) + 1e-10))

    @staticmethod
    def _hash_file(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
