"""Bridge between loguru and PyQt5: emits log messages as Qt signals."""

from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal


class QtLogHandler(QObject):
    """Loguru sink that forwards log records to a Qt signal.

    Usage::

        handler = QtLogHandler()
        handler.log_received.connect(some_slot)
        logger.add(handler.write, format="...", level="INFO")
    """

    log_received = pyqtSignal(str, str)  # (level_name, formatted_message)

    def write(self, message) -> None:  # noqa: ANN001 – loguru passes a Message object
        """Loguru sink callable. *message* is a loguru ``Message`` with a ``.record`` dict."""
        record = message.record
        level = record["level"].name
        text = str(message).rstrip("\n")
        self.log_received.emit(level, text)
