"""Data types used throughout the subtitle extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpeechSegment:
    """A segment of audio that contains speech, detected by VAD."""

    start_time: float  # seconds
    end_time: float  # seconds

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class WordTimestamp:
    """A single word with its time boundaries from ASR."""

    word: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class TranscriptionSegment:
    """ASR output for a speech segment: text with word-level timestamps."""

    start_time: float  # seconds
    end_time: float  # seconds
    text: str
    words: list[WordTimestamp] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SubtitleEntry:
    """A single subtitle line ready for export to CSV/SRT."""

    index: int
    start_time: float  # seconds
    end_time: float  # seconds
    text: str
    speaker: Optional[str] = None  # "A", "B", "C", ...

    def format_timestamp_srt(self, seconds: float) -> str:
        """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_srt_block(self) -> str:
        """Format as a single SRT subtitle block."""
        start_ts = self.format_timestamp_srt(self.start_time)
        end_ts = self.format_timestamp_srt(self.end_time)
        prefix = f"[{self.speaker}] " if self.speaker else ""
        return f"{self.index}\n{start_ts} --> {end_ts}\n{prefix}{self.text}\n"

    def to_csv_row(self) -> list[str]:
        """Return a list of fields for CSV export."""
        return [
            f"{self.start_time:.3f}",
            f"{self.end_time:.3f}",
            self.speaker or "",
            self.text,
        ]


class CancelledException(Exception):
    """Raised when the user cancels the processing pipeline."""

    pass
