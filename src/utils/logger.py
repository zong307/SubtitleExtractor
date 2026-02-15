"""Loguru logging setup with console, file, and optional UI sinks."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from loguru import logger


def _get_log_dir() -> Path:
    """Return platform-appropriate log directory."""
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        return Path(base) / "subtitle_extractor" / "logs"
    elif system == "Darwin":
        return Path.home() / "Library" / "Logs" / "subtitle_extractor"
    else:
        data_home = os.environ.get(
            "XDG_DATA_HOME", Path.home() / ".local" / "share"
        )
        return Path(data_home) / "subtitle_extractor" / "logs"


def setup_logger(log_level: str = "INFO") -> None:
    """Configure loguru with console and rotating file sinks.

    Call this once at application startup. The UI sink is added separately
    via ``add_ui_sink`` after the Qt log handler is created.
    """
    # Remove the default stderr sink so we can add our own
    logger.remove()

    # Console sink (coloured)
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <4}</level> | "
            "<cyan>{file}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Rotating file sink
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_dir / "app_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {file}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )


def add_ui_sink(sink_callable, log_level: str = "INFO") -> int:
    """Add a UI sink (e.g. QtLogHandler.write) to loguru.

    Returns the sink id so it can be removed later if needed.
    """
    return logger.add(
        sink_callable,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        colorize=False,
    )
