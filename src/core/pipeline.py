"""Main processing pipeline that orchestrates all subtitle extraction steps."""

from __future__ import annotations

import os
import shutil
import threading
from typing import Callable, Optional

from loguru import logger

from src.asr import ASRRegistry
from src.core.audio_processor import AudioProcessor
from src.core.speaker_diarizer import SpeakerDiarizer
from src.core.subtitle_generator import SubtitleGenerator
from src.core.vad_processor import VADProcessor
from src.models.datatypes import (
    CancelledException,
    SubtitleEntry,
    TranscriptionSegment,
)

# Type alias for the progress callback: (step_name, percent_0_100, detail_message)
ProgressCallback = Callable[[str, float, str], None]


class SubtitlePipeline:
    """End-to-end subtitle extraction pipeline.

    Orchestrates: audio extraction → enhancement → VAD → ASR → diarization → export.
    Supports cancellation with partial-result saving.
    """

    def __init__(
        self,
        config: dict,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self._config = config
        self._progress = progress_callback or (lambda *_: None)
        self._cancelled = threading.Event()
        self._partial_results: list[SubtitleEntry] = []
        self._temp_files: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: str,
        csv_output: str,
        srt_output: str,
    ) -> bool:
        """Run the full pipeline. Returns True on success, False on cancellation."""
        try:
            return self._run(input_path, csv_output, srt_output)
        except CancelledException:
            logger.warning("Pipeline cancelled by user.")
            self._save_partial(csv_output)
            return False
        except Exception:
            logger.exception("Pipeline failed with an unexpected error.")
            self._save_partial(csv_output)
            raise
        finally:
            self._cleanup()

    def cancel(self) -> None:
        """Signal the pipeline to stop at the next checkpoint."""
        logger.info("Cancel requested.")
        self._cancelled.set()

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _run(
        self,
        input_path: str,
        csv_output: str,
        srt_output: str,
    ) -> bool:
        audio_cfg = self._config.get("audio", {})
        vad_cfg = self._config.get("vad", {})
        asr_cfg = self._config.get("asr", {})
        sub_cfg = self._config.get("subtitle", {})
        diar_cfg = self._config.get("diarization", {})
        paths_cfg = self._config.get("paths", {})

        # Use project-local temp/ directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tmp_base = os.path.join(project_root, "temp")
        os.makedirs(tmp_base, exist_ok=True)
        # Create a unique sub-directory for this run
        import time
        tmp_dir = os.path.join(tmp_base, f"run_{int(time.time() * 1000)}")
        os.makedirs(tmp_dir, exist_ok=True)

        # -- Step 1: Extract audio ----------------------------------------
        self._check_cancelled()
        self._progress("extract", 0, "正在提取音频...")
        ap = AudioProcessor()
        raw_audio = os.path.join(tmp_dir, "raw_audio.wav")
        self._temp_files.append(raw_audio)
        ap.extract_audio(input_path, raw_audio)
        self._progress("extract", 5, "音频提取完成")

        # -- Step 2: Enhance audio ----------------------------------------
        self._check_cancelled()
        self._progress("enhance", 5, "正在增强人声...")
        enhanced_audio = os.path.join(tmp_dir, "enhanced.wav")
        self._temp_files.append(enhanced_audio)
        ap.enhance_audio(
            raw_audio,
            enhanced_audio,
            enable_noise_reduce=audio_cfg.get("enable_noise_reduce", True),
            enable_bandpass=audio_cfg.get("enable_bandpass", True),
        )
        audio_duration = ap.get_duration(enhanced_audio)
        self._progress("enhance", 10, "音频增强完成")

        # -- Step 3: VAD ---------------------------------------------------
        self._check_cancelled()
        self._progress("vad", 10, "正在检测语音片段...")
        vad = VADProcessor(
            threshold=vad_cfg.get("threshold", 0.5),
            min_silence_duration_ms=vad_cfg.get("min_silence_duration_ms", 300),
            min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 250),
            speech_pad_ms=vad_cfg.get("speech_pad_ms", 30),
        )
        raw_segments = vad.detect_speech(enhanced_audio)
        self._progress("vad", 15, f"检测到 {len(raw_segments)} 个语音片段")

        # -- Step 4: Expand + merge segments --------------------------------
        self._check_cancelled()
        padding_s = audio_cfg.get("padding_seconds", 2.0)
        segments = VADProcessor.expand_and_merge(raw_segments, padding_s, audio_duration)
        self._progress("vad", 18, f"合并后 {len(segments)} 个语音片段")

        # -- Step 4b: Split segments exceeding max duration -------------------
        max_speech_s = vad_cfg.get("max_speech_duration_s", 30.0)
        if max_speech_s > 0:
            segments = VADProcessor.split_long_segments(segments, max_speech_s)
        self._progress("vad", 20, f"最终 {len(segments)} 个语音片段")

        if not segments:
            logger.warning("No speech segments detected. Output files will be empty.")
            gen = SubtitleGenerator()
            gen.export_csv([], csv_output)
            gen.export_srt([], srt_output)
            self._progress("done", 100, "未检测到语音")
            return True

        # -- Step 5: ASR transcription per segment --------------------------
        self._check_cancelled()
        self._progress("asr", 20, "正在加载 ASR 模型...")

        model_dir = paths_cfg.get("model_dir", "") or None
        asr_engine = ASRRegistry.create(
            asr_type=asr_cfg.get("type", "whisper"),
            model_size=asr_cfg.get("model_size", "turbo"),
            device=asr_cfg.get("device", "cpu"),
            model_dir=model_dir,
        )
        asr_engine.load_model()
        self._progress("asr", 25, "ASR 模型加载完成，开始识别...")

        language = asr_cfg.get("language", "zh")
        all_transcriptions: list[TranscriptionSegment] = []
        asr_start_pct = 25.0
        asr_end_pct = 85.0

        for i, seg in enumerate(segments):
            self._check_cancelled()
            pct = asr_start_pct + (asr_end_pct - asr_start_pct) * (i / len(segments))
            self._progress(
                "asr",
                pct,
                f"正在识别语音片段 {i + 1}/{len(segments)}...",
            )

            # Extract the segment audio to a temp file
            seg_audio = os.path.join(tmp_dir, f"seg_{i}.wav")
            self._temp_files.append(seg_audio)
            self._extract_segment(enhanced_audio, seg_audio, seg.start_time, seg.end_time)

            trans = asr_engine.transcribe(seg_audio, language=language)

            # Offset timestamps by segment start_time (ASR timestamps are relative to segment)
            for t in trans:
                t.start_time += seg.start_time
                t.end_time += seg.start_time
                for w in t.words:
                    w.start += seg.start_time
                    w.end += seg.start_time

            all_transcriptions.extend(trans)

            # Update partial results (for cancel-safety)
            max_chars = sub_cfg.get("max_chars_per_subtitle", 30)
            gen_tmp = SubtitleGenerator(max_chars=max_chars)
            self._partial_results = gen_tmp.generate(all_transcriptions)

        asr_engine.unload_model()
        self._progress("asr", 85, f"ASR 识别完成，共 {len(all_transcriptions)} 个片段")

        # -- Step 6: Speaker diarization (optional) -------------------------
        speaker_map: Optional[dict[int, str]] = None
        if diar_cfg.get("enabled", False):
            self._check_cancelled()
            self._progress("diarization", 85, "正在进行说话人分离...")
            try:
                diarizer = SpeakerDiarizer()
                speaker_map = diarizer.identify_speakers(enhanced_audio, segments)
                self._progress("diarization", 92, "说话人分离完成")
            except Exception as e:
                logger.error(f"Speaker diarization failed: {e}. Continuing without it.")
                self._progress("diarization", 92, "说话人分离失败，继续处理")

        # -- Step 7: Generate subtitles + export ----------------------------
        self._check_cancelled()
        self._progress("export", 92, "正在生成字幕文件...")
        max_chars = sub_cfg.get("max_chars_per_subtitle", 30)
        gen = SubtitleGenerator(max_chars=max_chars)
        entries = gen.generate(all_transcriptions, speaker_map=speaker_map)

        gen.export_csv(entries, csv_output)
        gen.export_srt(entries, srt_output)

        self._progress("done", 100, f"完成！共生成 {len(entries)} 条字幕")
        logger.info(f"CSV saved to: {csv_output}")
        logger.info(f"SRT saved to: {srt_output}")
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise CancelledException("User cancelled the operation.")

    def _save_partial(self, csv_output: str) -> None:
        """Save whatever results we have so far."""
        if self._partial_results:
            try:
                SubtitleGenerator.export_csv(self._partial_results, csv_output)
                logger.info(
                    f"Partial results ({len(self._partial_results)} entries) "
                    f"saved to: {csv_output}"
                )
            except Exception as e:
                logger.error(f"Failed to save partial results: {e}")
        else:
            logger.info("No partial results to save.")

    @staticmethod
    def _extract_segment(
        audio_path: str,
        output_path: str,
        start_s: float,
        end_s: float,
    ) -> None:
        """Extract a time range from an audio file using soundfile."""
        import soundfile as sf

        info = sf.info(audio_path)
        sr = info.samplerate
        start_frame = int(start_s * sr)
        num_frames = int((end_s - start_s) * sr)
        data, _ = sf.read(audio_path, start=start_frame, frames=num_frames, dtype="float32")
        sf.write(output_path, data, sr, subtype="PCM_16")

    def _cleanup(self) -> None:
        """Remove temporary files and directories."""
        tmp_dirs: set[str] = set()
        for path in self._temp_files:
            tmp_dirs.add(os.path.dirname(path))
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()
        # Remove temp run directories
        for d in tmp_dirs:
            try:
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
