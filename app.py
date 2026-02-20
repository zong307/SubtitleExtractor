"""Subtitle Extractor – application entry point."""

import os
import sys

from src.config.settings import SettingsManager
from src.utils.logger import setup_logger


def main() -> None:
    # Load settings and set HuggingFace endpoint early, before importing any HF-related libraries
    settings = SettingsManager()
    hf_endpoint = settings.get("huggingface.endpoint", "")
    if hf_endpoint:
        os.environ['HF_ENDPOINT'] = hf_endpoint
        print(f"Using HuggingFace endpoint: {hf_endpoint}")  # Print to console for visibility
    else:
        # Remove the environment variable if it exists
        if 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']

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