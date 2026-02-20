"""Abstract base class for ASR (Automatic Speech Recognition) models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.models.datatypes import TranscriptionSegment


class ASRBase(ABC):
    """Interface that every ASR backend must implement.

    To add a new ASR engine:
        1. Create a subclass of ``ASRBase``.
        2. Implement all abstract methods.
        3. Register the class via ``ASRRegistry.register(YourClass)``.
    """

    def __init__(
        self,
        model_size: str,
        device: str = "cpu",
        model_dir: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.model_dir = model_dir
        self.config = config or {}
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the ASR model into memory."""
        ...

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> list[TranscriptionSegment]:
        """Transcribe an audio file and return segments with word timestamps.

        Args:
            audio_path: path to a 16 kHz mono WAV file.
            language: ISO language code (e.g. "zh", "en") or None for auto-detect.

        Returns:
            A list of ``TranscriptionSegment`` with word-level timestamps.
        """
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Release model resources."""
        ...

    @classmethod
    @abstractmethod
    def available_model_sizes(cls) -> list[str]:
        """Return the list of model-size identifiers this engine supports."""
        ...

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Human-readable name shown in the UI (e.g. 'Whisper')."""
        ...

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
