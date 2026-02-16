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

    def _align_words_with_text(self, full_text: str, raw_words: list[dict]) -> list[WordTimestamp]:
        """Align raw word timestamps with the actual text positions to preserve spacing."""
        if not full_text or not raw_words:
            return []
        
        # Sort raw words by their start time to maintain chronological order
        sorted_raw_words = sorted(raw_words, key=lambda x: x["start"])
        
        aligned_words: list[WordTimestamp] = []
        
        # Track our position in the original text
        text_idx = 0
        
        for raw_word_data in sorted_raw_words:
            word_text = raw_word_data["word"]
            start_time = raw_word_data["start"]
            end_time = raw_word_data["end"]
            
            if not word_text:
                continue
            
            # Find the word in the original text starting from our current position
            # This handles repeated words by finding the next occurrence
            word_pos = full_text.find(word_text, text_idx)
            
            if word_pos != -1:
                # The key improvement: we need to capture the actual text from the current
                # position in the text to the position of this word, which may include spaces
                if word_pos > text_idx:
                    # There's text between the current position and the found word
                    # This text likely contains spaces or punctuation that should be preserved with the previous word
                    spacing_text = full_text[text_idx:word_pos]
                    # We append this spacing to the previous word
                    if aligned_words:
                        # Append the spacing to the previous word's text
                        prev_word = aligned_words[-1]
                        aligned_words[-1] = WordTimestamp(
                            word=prev_word.word + spacing_text,
                            start=prev_word.start,
                            end=prev_word.end
                        )
                
                # Now add the actual word with its timestamps
                actual_word_text = full_text[word_pos:word_pos + len(word_text)]
                aligned_words.append(WordTimestamp(
                    word=actual_word_text,
                    start=start_time,
                    end=end_time
                ))
                
                # Move the text position forward past this word
                text_idx = word_pos + len(word_text)
            else:
                # Word not found in the remaining text, add it without precise positioning
                aligned_words.append(WordTimestamp(
                    word=word_text,
                    start=start_time,
                    end=end_time
                ))
        
        # Handle any remaining text after the last word (e.g., punctuation at the end)
        if text_idx < len(full_text) and aligned_words:
            # Append any remaining text (like punctuation) to the last word
            remaining_text = full_text[text_idx:]
            last_word = aligned_words[-1]
            aligned_words[-1] = WordTimestamp(
                word=last_word.word + remaining_text,
                start=last_word.start,
                end=last_word.end
            )
        
        return aligned_words

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
            text = text.strip("\n")
            if not text:
                continue

            # Extract word-level timestamps if available
            words: list[WordTimestamp] = []
            raw_timestamps = getattr(res, "time_stamps", None) or getattr(
                res, "words", None
            )
            if raw_timestamps and hasattr(raw_timestamps, "items"):
                # Process word-level timestamps to preserve spacing in the original text
                raw_words = []
                for item in raw_timestamps.items:
                    if isinstance(item, dict):
                        raw_words.append({
                            "word": item.get("word", item.get("text", "")).strip(),
                            "start": float(item.get("start_time", 0)),
                            "end": float(item.get("end_time", 0)),
                        })
                    elif hasattr(item, "word") or hasattr(item, "text"):
                        raw_words.append({
                            "word": getattr(item, "word", getattr(item, "text", "")).strip(),
                            "start": float(getattr(item, "start_time", 0)),
                            "end": float(getattr(item, "end_time", 0)),
                        })
                
                # Reconstruct the text with proper spacing by mapping words back to the original text
                if raw_words:
                    words = self._align_words_with_text(text, raw_words)
                else:
                    words = []

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