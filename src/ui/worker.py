"""QThread worker that runs the subtitle extraction pipeline in the background."""

from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal
from loguru import logger

from src.core.pipeline import SubtitlePipeline
from src.models.datatypes import CancelledException


class PipelineWorker(QThread):
    """Runs SubtitlePipeline.process() on a background thread.

    Signals
    -------
    progress_updated(str, float, str)
        (step_name, percent, detail_message)
    finished_ok(str)
        Emitted on successful completion with a summary message.
    finished_error(str)
        Emitted when the pipeline fails with an error description.
    finished_cancelled(str)
        Emitted when the user cancels, with a message about partial saves.
    """

    progress_updated = pyqtSignal(str, float, str)
    finished_ok = pyqtSignal(str)
    finished_error = pyqtSignal(str)
    finished_cancelled = pyqtSignal(str)

    def __init__(
        self,
        config: dict,
        input_path: str,
        csv_path: str,
        srt_path: str,
    ) -> None:
        super().__init__()
        self._config = config
        self._input_path = input_path
        self._csv_path = csv_path
        self._srt_path = srt_path
        self._pipeline: SubtitlePipeline | None = None

    def run(self) -> None:
        """Entry point executed on the worker thread."""
        def _on_progress(step: str, pct: float, msg: str) -> None:
            self.progress_updated.emit(step, pct, msg)

        self._pipeline = SubtitlePipeline(
            config=self._config,
            progress_callback=_on_progress,
        )

        try:
            ok = self._pipeline.process(
                self._input_path, self._csv_path, self._srt_path
            )
            if ok:
                self.finished_ok.emit(
                    f"处理完成！字幕已保存至:\n  CSV: {self._csv_path}\n  SRT: {self._srt_path}"
                )
            else:
                self.finished_cancelled.emit(
                    f"处理已取消。部分结果已保存至:\n  {self._csv_path}"
                )
        except CancelledException:
            self.finished_cancelled.emit(
                f"处理已取消。部分结果已保存至:\n  {self._csv_path}"
            )
        except Exception as e:
            logger.exception("Pipeline worker encountered an error.")
            self.finished_error.emit(str(e))

    def cancel(self) -> None:
        """Request the pipeline to stop."""
        if self._pipeline is not None:
            self._pipeline.cancel()
