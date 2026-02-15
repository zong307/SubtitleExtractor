"""Qwen3-ASR backend."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from src.asr.base import ASRBase
from src.asr.registry import ASRRegistry
from src.models.datatypes import TranscriptionSegment, WordTimestamp


# Language code mapping for Qwen3-ASR (expects full language names)
_LANG_MAP: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "auto": None,  # type: ignore[dict-item]
}

# Model size -> HuggingFace model id
_MODEL_MAP: dict[str, str] = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
}

_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"


@ASRRegistry.register
class QwenASR(ASRBase):
    """ASR backend using Qwen3-ASR from Alibaba."""

    def __init__(self, model_size: str, device: str = "cpu", model_dir: str | None = None) -> None:
        super().__init__(model_size, device, model_dir)
        self._has_aligner = False

    @classmethod
    def name(cls) -> str:
        return "Qwen3-ASR"

    @classmethod
    def available_model_sizes(cls) -> list[str]:
        return list(_MODEL_MAP.keys())

    def load_model(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        model_id = _MODEL_MAP.get(self.model_size)
        if model_id is None:
            raise ValueError(
                f"Unknown Qwen3-ASR model size '{self.model_size}'. "
                f"Available: {list(_MODEL_MAP.keys())}"
            )

        # Select dtype based on device
        if self.device == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        logger.info(f"Loading Qwen3-ASR model '{model_id}' on {self.device}...")
        kwargs: dict = {
            "dtype": dtype,
            "device_map": self.device,
        }
        if self.model_dir:
            kwargs["cache_dir"] = self.model_dir

        # Load with forced aligner for word-level timestamps
        aligner_kwargs: dict = {"dtype": dtype, "device_map": self.device}
        if self.model_dir:
            aligner_kwargs["cache_dir"] = self.model_dir

        try:
            self._model = Qwen3ASRModel.from_pretrained(
                model_id,
                forced_aligner=_ALIGNER_MODEL,
                forced_aligner_kwargs=aligner_kwargs,
                **kwargs,
            )
            self._has_aligner = True
        except Exception as e:
            # Fall back without forced aligner if it fails
            logger.warning(
                f"Failed to load ForcedAligner ({e}); "
                "loading Qwen3-ASR without word-level timestamps."
            )
            self._model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
            self._has_aligner = False

        logger.info("Qwen3-ASR model loaded.")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> list[TranscriptionSegment]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        lang_name = _LANG_MAP.get(language) if language else None

        logger.debug(f"Qwen3-ASR transcribing: {audio_path}")

        transcribe_kwargs: dict = {
            "audio": audio_path,
            "language": lang_name,
        }
        if self._has_aligner:
            transcribe_kwargs["return_time_stamps"] = True

        results = self._model.transcribe(**transcribe_kwargs)

        segments: list[TranscriptionSegment] = []
        for res in results:
            text = getattr(res, "text", "") or ""
            text = text.strip()
            if not text:
                continue

            # Extract word-level timestamps if available
            words: list[WordTimestamp] = []
            raw_timestamps = getattr(res, "timestamps", None) or getattr(
                res, "words", None
            )
            if raw_timestamps:
                for item in raw_timestamps:
                    if isinstance(item, dict):
                        words.append(
                            WordTimestamp(
                                word=item.get("word", item.get("text", "")).strip(),
                                start=float(item.get("start", 0)),
                                end=float(item.get("end", 0)),
                            )
                        )
                    elif hasattr(item, "word") or hasattr(item, "text"):
                        words.append(
                            WordTimestamp(
                                word=getattr(
                                    item, "word", getattr(item, "text", "")
                                ).strip(),
                                start=float(getattr(item, "start", 0)),
                                end=float(getattr(item, "end", 0)),
                            )
                        )

            # Determine segment time range
            start_time = float(getattr(res, "start", 0))
            end_time = float(getattr(res, "end", 0))
            if words and start_time == 0 and end_time == 0:
                start_time = words[0].start
                end_time = words[-1].end

            segments.append(
                TranscriptionSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    words=words,
                )
            )

        logger.debug(f"Qwen3-ASR returned {len(segments)} segment(s).")
        return segments

    def unload_model(self) -> None:
        self._model = None
        # Attempt to free GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Qwen3-ASR model unloaded.")
