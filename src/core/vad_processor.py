"""Voice Activity Detection using Silero-VAD."""

from __future__ import annotations

import math

from loguru import logger

from src.models.datatypes import SpeechSegment


class VADProcessor:
    """Detect speech segments in audio using Silero-VAD, then pad and merge."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        min_speech_duration_ms: int = 250,
        speech_pad_ms: int = 30,
    ) -> None:
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._model = None

    def _ensure_model(self) -> None:
        """Lazy-load the Silero-VAD model."""
        if self._model is None:
            from silero_vad import load_silero_vad

            logger.info("Loading Silero-VAD model...")
            self._model = load_silero_vad()
            logger.info("Silero-VAD model loaded.")

    def detect_speech(self, audio_path: str) -> list[SpeechSegment]:
        """Run VAD on audio and return speech segments (in seconds)."""
        from silero_vad import get_speech_timestamps, read_audio

        self._ensure_model()

        logger.info(f"Running VAD on: {audio_path}")
        wav = read_audio(audio_path)

        timestamps = get_speech_timestamps(
            wav,
            self._model,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            min_speech_duration_ms=self.min_speech_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True,
        )

        segments = [
            SpeechSegment(start_time=ts["start"], end_time=ts["end"])
            for ts in timestamps
        ]
        logger.info(f"VAD detected {len(segments)} speech segment(s).")
        return segments

    @staticmethod
    def expand_and_merge(
        segments: list[SpeechSegment],
        padding_s: float,
        audio_duration: float,
    ) -> list[SpeechSegment]:
        """Expand each segment by *padding_s* seconds on both sides, then merge overlaps.

        Args:
            segments: speech segments from VAD.
            padding_s: seconds to pad before and after each segment.
            audio_duration: total audio length in seconds (for clamping).

        Returns:
            Merged speech segments.
        """
        if not segments:
            return []

        # Expand
        expanded = []
        for seg in segments:
            start = max(0.0, seg.start_time - padding_s)
            end = min(audio_duration, seg.end_time + padding_s)
            expanded.append(SpeechSegment(start_time=start, end_time=end))

        # Sort by start time
        expanded.sort(key=lambda s: s.start_time)

        # Merge overlapping
        merged: list[SpeechSegment] = [expanded[0]]
        for seg in expanded[1:]:
            last = merged[-1]
            if seg.start_time <= last.end_time:
                # Overlapping or adjacent – extend
                merged[-1] = SpeechSegment(
                    start_time=last.start_time,
                    end_time=max(last.end_time, seg.end_time),
                )
            else:
                merged.append(seg)

        logger.info(
            f"After padding ({padding_s}s) and merging: {len(merged)} segment(s) "
            f"(from {len(segments)} original)."
        )
        return merged

    @staticmethod
    def split_long_segments(
        segments: list[SpeechSegment],
        max_duration_s: float,
    ) -> list[SpeechSegment]:
        """Split segments that exceed *max_duration_s* into smaller chunks.

        Args:
            segments: speech segments to process.
            max_duration_s: maximum allowed duration per segment in seconds.

        Returns:
            List of segments, with any over-long segments split evenly.
        """
        if max_duration_s <= 0:
            return segments

        result: list[SpeechSegment] = []
        for seg in segments:
            duration = seg.end_time - seg.start_time
            if duration <= max_duration_s:
                result.append(seg)
            else:
                # Split into roughly equal chunks, each <= max_duration_s
                n_chunks = math.ceil(duration / max_duration_s)
                chunk_len = duration / n_chunks
                for j in range(n_chunks):
                    chunk_start = seg.start_time + j * chunk_len
                    chunk_end = seg.start_time + (j + 1) * chunk_len
                    if j == n_chunks - 1:
                        chunk_end = seg.end_time  # avoid floating-point drift
                    result.append(SpeechSegment(start_time=chunk_start, end_time=chunk_end))
                logger.info(
                    f"Split segment [{seg.start_time:.1f}s - {seg.end_time:.1f}s] "
                    f"({duration:.1f}s) into {n_chunks} chunks (max {max_duration_s}s each)"
                )
        if len(result) != len(segments):
            logger.info(
                f"After splitting long segments: {len(result)} segment(s) "
                f"(from {len(segments)})."
            )
        return result
