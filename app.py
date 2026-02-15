"""Subtitle Extractor – application entry point."""

import sys

from src.utils.logger import setup_logger


def main() -> None:
    setup_logger()

    from loguru import logger

    logger.info("Initializing Subtitle Extractor...")

    from PyQt5.QtWidgets import QApplication

    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Subtitle Extractor")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
