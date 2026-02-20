"""OpenAI Whisper ASR backend."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from src.asr.base import ASRBase
from src.asr.registry import ASRRegistry
from src.models.datatypes import TranscriptionSegment, WordTimestamp


@ASRRegistry.register
class WhisperASR(ASRBase):
    """ASR backend using OpenAI's open-source Whisper model."""

    _MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "turbo"]

    @classmethod
    def name(cls) -> str:
        return "Whisper"

    @classmethod
    def available_model_sizes(cls) -> list[str]:
        return list(cls._MODEL_SIZES)

    def load_model(self) -> None:
        import whisper
        import os
        
        # The HF_ENDPOINT environment variable should already be set at application startup
        # and when user changes the setting, so we just log the current value
        current_endpoint = os.environ.get('HF_ENDPOINT', 'Official HuggingFace endpoint')
        logger.info(f"Using HuggingFace endpoint: {current_endpoint}")

        logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}...")
        kwargs: dict = {"name": self.model_size, "device": self.device}
        if self.model_dir:
            kwargs["download_root"] = self.model_dir
        self._model = whisper.load_model(**kwargs)
        logger.info("Whisper model loaded.")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> list[TranscriptionSegment]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.debug(f"Whisper transcribing: {audio_path}")
        opts: dict = {"word_timestamps": True}
        if language and language != "auto":
            opts["language"] = language

        result = self._model.transcribe(audio_path, **opts)

        segments: list[TranscriptionSegment] = []
        for seg in result.get("segments", []):
            words: list[WordTimestamp] = []
            for w in seg.get("words", []):
                words.append(
                    WordTimestamp(
                        word=w.get("word", "").strip(),
                        start=float(w.get("start", 0)),
                        end=float(w.get("end", 0)),
                    )
                )
            segments.append(
                TranscriptionSegment(
                    start_time=float(seg["start"]),
                    end_time=float(seg["end"]),
                    text=seg.get("text", "").strip(),
                    words=words,
                )
            )
        logger.debug(f"Whisper returned {len(segments)} segment(s).")
        return segments

    def unload_model(self) -> None:
        self._model = None
        logger.info("Whisper model unloaded.")
