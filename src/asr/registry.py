"""ASR model registry and factory."""

from __future__ import annotations

from typing import Optional, Type

from src.asr.base import ASRBase


class ASRRegistry:
    """Central registry for all ASR backends.

    Use ``ASRRegistry.register(cls)`` to register new backends, and
    ``ASRRegistry.create(asr_type, ...)`` to instantiate them.
    """

    _registry: dict[str, Type[ASRBase]] = {}

    @classmethod
    def register(cls, asr_class: Type[ASRBase]) -> Type[ASRBase]:
        """Register an ASR class. Can also be used as a decorator."""
        key = asr_class.name().lower()
        cls._registry[key] = asr_class
        return asr_class

    @classmethod
    def create(
        cls,
        asr_type: str,
        model_size: str,
        device: str = "cpu",
        model_dir: Optional[str] = None,
    ) -> ASRBase:
        """Instantiate an ASR backend by its registered name."""
        key = asr_type.lower()
        if key not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown ASR type '{asr_type}'. Available: {available}"
            )
        return cls._registry[key](
            model_size=model_size,
            device=device,
            model_dir=model_dir,
        )

    @classmethod
    def list_types(cls) -> list[str]:
        """Return registered ASR type names."""
        return list(cls._registry.keys())

    @classmethod
    def get_model_sizes(cls, asr_type: str) -> list[str]:
        """Return available model sizes for a given ASR type."""
        key = asr_type.lower()
        if key not in cls._registry:
            return []
        return cls._registry[key].available_model_sizes()

    @classmethod
    def get_display_name(cls, asr_type: str) -> str:
        """Return the human-readable display name for an ASR type."""
        key = asr_type.lower()
        if key in cls._registry:
            return cls._registry[key].name()
        return asr_type
