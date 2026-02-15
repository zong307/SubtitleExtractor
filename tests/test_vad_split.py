"""Tests for VADProcessor.split_long_segments."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.datatypes import SpeechSegment
from src.core.vad_processor import VADProcessor


def test_no_split_needed():
    """Segments within max duration should remain unchanged."""
    segs = [
        SpeechSegment(start_time=0.0, end_time=10.0),
        SpeechSegment(start_time=15.0, end_time=25.0),
    ]
    result = VADProcessor.split_long_segments(segs, max_duration_s=30.0)
    assert len(result) == 2
    assert result[0].start_time == 0.0
    assert result[0].end_time == 10.0
    assert result[1].start_time == 15.0
    assert result[1].end_time == 25.0


def test_split_single_long_segment():
    """A 60s segment with max 30s should be split into 2 chunks."""
    segs = [SpeechSegment(start_time=0.0, end_time=60.0)]
    result = VADProcessor.split_long_segments(segs, max_duration_s=30.0)
    assert len(result) == 2
    assert result[0].start_time == 0.0
    assert abs(result[0].end_time - 30.0) < 0.01
    assert abs(result[1].start_time - 30.0) < 0.01
    assert result[1].end_time == 60.0


def test_split_very_long_segment():
    """A 100s segment with max 30s should be split into 4 chunks (~25s each)."""
    segs = [SpeechSegment(start_time=10.0, end_time=110.0)]
    result = VADProcessor.split_long_segments(segs, max_duration_s=30.0)
    # 100s / 30s = 3.33 -> 4 chunks
    assert len(result) == 4
    # All chunks should be <= 30s
    for seg in result:
        assert seg.end_time - seg.start_time <= 30.0 + 0.01
    # First chunk starts at original start
    assert result[0].start_time == 10.0
    # Last chunk ends at original end
    assert result[-1].end_time == 110.0
    # Chunks should be contiguous
    for i in range(len(result) - 1):
        assert abs(result[i].end_time - result[i + 1].start_time) < 0.01


def test_split_mixed_segments():
    """Mix of short and long segments."""
    segs = [
        SpeechSegment(start_time=0.0, end_time=5.0),    # short
        SpeechSegment(start_time=10.0, end_time=80.0),   # 70s -> split
        SpeechSegment(start_time=90.0, end_time=100.0),  # short
    ]
    result = VADProcessor.split_long_segments(segs, max_duration_s=30.0)
    # First stays, second split into 3 (70/30=2.33 -> 3 chunks), third stays
    assert result[0].start_time == 0.0
    assert result[0].end_time == 5.0
    assert result[-1].start_time == 90.0
    assert result[-1].end_time == 100.0
    # Middle segment split into 3
    assert len(result) == 5


def test_zero_max_duration_no_split():
    """max_duration_s <= 0 should return segments unchanged."""
    segs = [SpeechSegment(start_time=0.0, end_time=100.0)]
    result = VADProcessor.split_long_segments(segs, max_duration_s=0)
    assert len(result) == 1
    assert result[0].end_time == 100.0


def test_empty_segments():
    """Empty input should return empty output."""
    result = VADProcessor.split_long_segments([], max_duration_s=30.0)
    assert result == []


def test_exact_boundary():
    """Segment exactly at max duration should not be split."""
    segs = [SpeechSegment(start_time=0.0, end_time=30.0)]
    result = VADProcessor.split_long_segments(segs, max_duration_s=30.0)
    assert len(result) == 1


if __name__ == "__main__":
    test_no_split_needed()
    test_split_single_long_segment()
    test_split_very_long_segment()
    test_split_mixed_segments()
    test_zero_max_duration_no_split()
    test_empty_segments()
    test_exact_boundary()
    print("All tests passed!")
