"""Subtitle generation: sentence splitting, length enforcement, CSV/SRT export."""

from __future__ import annotations

import csv
import re
from typing import Optional

from loguru import logger

from src.models.datatypes import (
    SubtitleEntry,
    TranscriptionSegment,
    WordTimestamp,
)

# Sentence-ending punctuation (Chinese and Western)
_SENTENCE_END_RE = re.compile(r"([。！？.!?]+)")


class SubtitleGenerator:
    """Convert ASR transcription segments into properly split subtitle entries."""

    def __init__(self, max_chars: int = 30) -> None:
        self.max_chars = max_chars

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as readable timestamp: HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def generate(
        self,
        segments: list[TranscriptionSegment],
        speaker_map: Optional[dict[int, str]] = None,
    ) -> list[SubtitleEntry]:
        """Build subtitle entries from transcription segments.

        1. Split each segment at sentence-ending punctuation.
        2. Further split any piece exceeding *max_chars* at word boundaries.
        3. Assign speaker labels if *speaker_map* is provided.
        """
        entries: list[SubtitleEntry] = []
        index = 1

        for seg_idx, seg in enumerate(segments):
            speaker = speaker_map.get(seg_idx) if speaker_map else None

            # Step 1: split by sentences
            sentence_parts = self._split_by_sentences(seg)

            for part in sentence_parts:
                # Step 2: split long sentences
                if len(part.text) > self.max_chars and part.words:
                    sub_parts = self._split_long_segment(part, self.max_chars)
                else:
                    sub_parts = [part]

                for sp in sub_parts:
                    if not sp.text.strip():
                        continue
                    entry = SubtitleEntry(
                        index=index,
                        start_time=sp.start_time,
                        end_time=sp.end_time,
                        text=sp.text.strip(),
                        speaker=speaker,
                    )
                    entries.append(entry)
                    # Log the subtitle with timestamp and content
                    start_ts = self._format_timestamp(entry.start_time)
                    end_ts = self._format_timestamp(entry.end_time)
                    speaker_prefix = f"[{entry.speaker}] " if entry.speaker else ""
                    logger.info(f"[{start_ts} --> {end_ts}] {speaker_prefix}{entry.text}")
                    index += 1

        logger.info(f"Generated {len(entries)} subtitle entries.")
        return entries

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    def _split_by_sentences(
        self, segment: TranscriptionSegment
    ) -> list[TranscriptionSegment]:
        """Split a segment at sentence-ending punctuation using word timestamps."""
        text = segment.text
        if not text:
            return []

        # Find split points
        parts = _SENTENCE_END_RE.split(text)
        # Rejoin: ['今天天气很好', '。', '要不要出去玩', '？', '']
        # -> ['今天天气很好。', '要不要出去玩？']
        sentences: list[str] = []
        i = 0
        while i < len(parts):
            chunk = parts[i]
            # Attach trailing punctuation
            if i + 1 < len(parts) and _SENTENCE_END_RE.fullmatch(parts[i + 1]):
                chunk += parts[i + 1]
                i += 2
            else:
                i += 1
            chunk = chunk.strip()
            if chunk:
                sentences.append(chunk)

        if len(sentences) <= 1:
            return [segment]

        # Assign timestamps using word-level data
        if not segment.words:
            # No word timestamps – distribute time proportionally
            return self._split_by_proportion(segment, sentences)

        return self._assign_timestamps_to_sentences(segment, sentences)

    def _assign_timestamps_to_sentences(
        self,
        segment: TranscriptionSegment,
        sentences: list[str],
    ) -> list[TranscriptionSegment]:
        """Map sentence strings back to word timestamps for accurate time boundaries."""
        results: list[TranscriptionSegment] = []
        words = list(segment.words)
        word_idx = 0

        for sentence in sentences:
            # Collect words that belong to this sentence
            sentence_words: list[WordTimestamp] = []
            remaining = sentence

            while word_idx < len(words) and remaining:
                w = words[word_idx]
                clean_word = w.word.strip()
                if not clean_word:
                    word_idx += 1
                    continue

                # Check if this word appears in the remaining text
                pos = remaining.find(clean_word)
                if pos == -1:
                    # Try removing punctuation for matching
                    stripped = clean_word.rstrip("。！？.!?,，、;；:")
                    pos = remaining.find(stripped) if stripped else -1

                if pos != -1:
                    sentence_words.append(w)
                    # Advance remaining past this word
                    end_pos = pos + len(clean_word)
                    remaining = remaining[end_pos:].lstrip()
                    word_idx += 1
                else:
                    # Word doesn't match – it likely belongs to the next sentence
                    break

            if sentence_words:
                start = sentence_words[0].start
                end = sentence_words[-1].end
            elif results:
                start = results[-1].end_time
                end = start + 0.5
            else:
                start = segment.start_time
                end = start + 0.5

            results.append(
                TranscriptionSegment(
                    start_time=start,
                    end_time=end,
                    text=sentence,
                    words=sentence_words,
                )
            )

        return results

    @staticmethod
    def _split_by_proportion(
        segment: TranscriptionSegment,
        sentences: list[str],
    ) -> list[TranscriptionSegment]:
        """Fallback: distribute time proportionally by character count."""
        total_chars = sum(len(s) for s in sentences)
        if total_chars == 0:
            return [segment]

        results: list[TranscriptionSegment] = []
        current_time = segment.start_time
        duration = segment.duration

        for sentence in sentences:
            ratio = len(sentence) / total_chars
            end_time = current_time + duration * ratio
            results.append(
                TranscriptionSegment(
                    start_time=current_time,
                    end_time=end_time,
                    text=sentence,
                    words=[],
                )
            )
            current_time = end_time

        return results

    # ------------------------------------------------------------------
    # Long-sentence splitting
    # ------------------------------------------------------------------

    def _split_long_segment(
        self,
        segment: TranscriptionSegment,
        max_chars: int,
    ) -> list[TranscriptionSegment]:
        """Split a long segment at word boundaries respecting max_chars."""
        words = segment.words
        if not words:
            # No word timestamps – split text evenly
            return self._split_text_evenly(segment, max_chars)

        results: list[TranscriptionSegment] = []
        current_words: list[WordTimestamp] = []
        current_text = ""

        for w in words:
            candidate = current_text + w.word
            if len(candidate.strip()) > max_chars and current_words:
                results.append(
                    TranscriptionSegment(
                        start_time=current_words[0].start,
                        end_time=current_words[-1].end,
                        text=current_text.strip(),
                        words=list(current_words),
                    )
                )
                current_words = [w]
                current_text = w.word
            else:
                current_words.append(w)
                current_text = candidate

        if current_words:
            results.append(
                TranscriptionSegment(
                    start_time=current_words[0].start,
                    end_time=current_words[-1].end,
                    text=current_text.strip(),
                    words=list(current_words),
                )
            )

        return results if results else [segment]

    @staticmethod
    def _split_text_evenly(
        segment: TranscriptionSegment,
        max_chars: int,
    ) -> list[TranscriptionSegment]:
        """Split text into chunks of at most max_chars, distributing time evenly."""
        text = segment.text
        chunks: list[str] = []
        while len(text) > max_chars:
            # Try to find a natural break (comma, space)
            split_pos = max_chars
            for sep in (",", "，", " ", "、"):
                pos = text.rfind(sep, 0, max_chars)
                if pos > max_chars // 2:
                    split_pos = pos + 1
                    break
            chunks.append(text[:split_pos].strip())
            text = text[split_pos:].strip()
        if text:
            chunks.append(text)

        if not chunks:
            return [segment]

        duration = segment.duration
        per_chunk = duration / len(chunks) if chunks else duration
        results: list[TranscriptionSegment] = []
        t = segment.start_time
        for chunk in chunks:
            results.append(
                TranscriptionSegment(
                    start_time=t,
                    end_time=t + per_chunk,
                    text=chunk,
                    words=[],
                )
            )
            t += per_chunk
        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_csv(entries: list[SubtitleEntry], path: str) -> None:
        """Write subtitle entries to a CSV file."""
        logger.info(f"Exporting {len(entries)} entries to CSV: {path}")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["start_time", "end_time", "speaker", "text"])
            for entry in entries:
                writer.writerow(entry.to_csv_row())

    @staticmethod
    def export_srt(entries: list[SubtitleEntry], path: str) -> None:
        """Write subtitle entries to an SRT file."""
        logger.info(f"Exporting {len(entries)} entries to SRT: {path}")
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(entry.to_srt_block())
                f.write("\n")
