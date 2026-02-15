"""ASR package – import submodules to trigger auto-registration."""

from src.asr.base import ASRBase  # noqa: F401
from src.asr.registry import ASRRegistry  # noqa: F401

# Import backends so they register themselves via @ASRRegistry.register
from src.asr import whisper_asr as _whisper  # noqa: F401
from src.asr import qwen_asr as _qwen  # noqa: F401
