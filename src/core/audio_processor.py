"""Audio extraction from video and voice enhancement."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from loguru import logger
from scipy.signal import butter, sosfilt


class AudioProcessor:
    """Handles audio extraction from media files and human-voice enhancement."""

    def extract_audio(
        self,
        input_path: str,
        output_path: str,
        sample_rate: int = 16000,
    ) -> str:
        """Extract audio from a video/audio file as 16 kHz mono WAV.

        Uses ffmpeg-python. Both video and audio inputs are supported.
        Returns the output path.
        """
        import ffmpeg

        logger.info(f"Extracting audio from: {input_path}")
        try:
            (
                ffmpeg.input(input_path)
                .output(
                    output_path,
                    ar=sample_rate,
                    ac=1,
                    format="wav",
                    acodec="pcm_s16le",
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"ffmpeg extraction failed: {e.stderr}")
            raise RuntimeError(
                f"Failed to extract audio from '{input_path}'. "
                "Ensure ffmpeg is installed and the file is a valid media file."
            ) from e
        logger.info(f"Audio extracted to: {output_path}")
        return output_path

    def enhance_audio(
        self,
        audio_path: str,
        output_path: str,
        enable_noise_reduce: bool = True,
        enable_bandpass: bool = True,
    ) -> str:
        """Enhance human voice: noise reduction + bandpass filter + normalization.

        Returns the output path.
        """
        logger.info("Enhancing audio (voice enhancement)...")
        data, sr = sf.read(audio_path, dtype="float32")

        if enable_noise_reduce:
            data = self._reduce_noise(data, sr)

        if enable_bandpass:
            data = self._bandpass_filter(data, sr, lowcut=80, highcut=8000)

        data = self._normalize(data)

        sf.write(output_path, data, sr, subtype="PCM_16")
        logger.info(f"Enhanced audio saved to: {output_path}")
        return output_path

    def get_duration(self, audio_path: str) -> float:
        """Return the duration of an audio file in seconds."""
        info = sf.info(audio_path)
        return info.duration

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_noise(data: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral noise reduction using noisereduce."""
        import noisereduce as nr

        logger.debug("Applying noise reduction...")
        return nr.reduce_noise(y=data, sr=sr, prop_decrease=0.8)

    @staticmethod
    def _bandpass_filter(
        data: np.ndarray,
        sr: int,
        lowcut: int = 80,
        highcut: int = 8000,
        order: int = 5,
    ) -> np.ndarray:
        """Apply a Butterworth bandpass filter to keep human-voice frequencies."""
        logger.debug(f"Applying bandpass filter ({lowcut}-{highcut} Hz)...")
        nyquist = sr / 2.0
        low = lowcut / nyquist
        high = min(highcut / nyquist, 0.99)  # clamp to avoid exceeding Nyquist
        sos = butter(order, [low, high], btype="band", output="sos")
        return sosfilt(sos, data).astype(np.float32)

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """Normalize audio to peak amplitude of 0.95."""
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data * (0.95 / peak)
        return data
