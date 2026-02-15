"""Dark theme QSS stylesheet for the subtitle extractor UI."""

from __future__ import annotations


def get_dark_stylesheet() -> str:
    """Return a complete QSS string for the modern dark theme."""
    return """
/* ================================================================
   Global
   ================================================================ */
QWidget {
    background-color: #1E1E2E;
    color: #E0E0E0;
    font-family: "Segoe UI", "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    font-size: 10pt;
}

/* ================================================================
   QGroupBox
   ================================================================ */
QGroupBox {
    background-color: #252538;
    border: 1px solid #3A3A4E;
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    color: #B0B0C0;
    font-size: 10pt;
}

/* ================================================================
   QLabel
   ================================================================ */
QLabel {
    background: transparent;
    color: #E0E0E0;
}

/* ================================================================
   QLineEdit / QSpinBox / QDoubleSpinBox
   ================================================================ */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1A1A2E;
    border: 1px solid #3A3A4E;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
    color: #E0E0E0;
    selection-background-color: #6366F1;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #6366F1;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #16162A;
    color: #606070;
}

/* spin box buttons */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #252538;
    border: none;
    width: 18px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #3A3A4E;
}

/* ================================================================
   QComboBox
   ================================================================ */
QComboBox {
    background-color: #1A1A2E;
    border: 1px solid #3A3A4E;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
    color: #E0E0E0;
}
QComboBox:focus {
    border: 1px solid #6366F1;
}
QComboBox:disabled {
    background-color: #16162A;
    color: #606070;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #B0B0C0;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #252538;
    border: 1px solid #3A3A4E;
    selection-background-color: #6366F1;
    color: #E0E0E0;
}

/* ================================================================
   QPushButton
   ================================================================ */
QPushButton {
    background-color: #252538;
    border: 1px solid #3A3A4E;
    border-radius: 4px;
    padding: 6px 16px;
    min-height: 24px;
    color: #E0E0E0;
}
QPushButton:hover {
    background-color: #2A2A42;
    border: 1px solid #4A4A5E;
}
QPushButton:pressed {
    background-color: #1E1E30;
}
QPushButton:disabled {
    background-color: #1A1A2E;
    color: #505060;
    border: 1px solid #2A2A3E;
}

/* Primary action button */
QPushButton#start_btn {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366F1, stop:1 #8B5CF6);
    border: none;
    border-radius: 6px;
    padding: 8px 32px;
    min-height: 32px;
    color: #FFFFFF;
    font-weight: bold;
    font-size: 11pt;
}
QPushButton#start_btn:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7578F5, stop:1 #9D75F8);
}
QPushButton#start_btn:pressed {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #5558E0, stop:1 #7B4CE6);
}
QPushButton#start_btn:disabled {
    background-color: #3A3A4E;
    color: #606070;
}

/* Stop / danger button */
QPushButton#stop_btn {
    background-color: transparent;
    border: 1px solid #EF4444;
    color: #EF4444;
    border-radius: 6px;
    padding: 8px 24px;
    min-height: 32px;
    font-weight: bold;
}
QPushButton#stop_btn:hover {
    background-color: #EF4444;
    color: #FFFFFF;
}
QPushButton#stop_btn:disabled {
    border: 1px solid #3A3A4E;
    color: #505060;
}

/* ================================================================
   QCheckBox / QRadioButton
   ================================================================ */
QCheckBox, QRadioButton {
    background: transparent;
    spacing: 6px;
    color: #E0E0E0;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #3A3A4E;
    background-color: #1A1A2E;
}
QCheckBox::indicator {
    border-radius: 3px;
}
QRadioButton::indicator {
    border-radius: 8px;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #6366F1;
    border: 1px solid #6366F1;
}
QCheckBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled {
    background-color: #3A3A4E;
    border: 1px solid #3A3A4E;
}

/* ================================================================
   QProgressBar
   ================================================================ */
QProgressBar {
    background-color: #1A1A2E;
    border: 1px solid #3A3A4E;
    border-radius: 4px;
    min-height: 22px;
    text-align: center;
    color: #FFFFFF;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366F1, stop:1 #8B5CF6);
    border-radius: 3px;
}

/* ================================================================
   QTextEdit (log viewer)
   ================================================================ */
QTextEdit#log_viewer {
    background-color: #14142A;
    border: 1px solid #3A3A4E;
    border-radius: 4px;
    font-family: "Cascadia Code", "Consolas", "Monaco", "Courier New", monospace;
    font-size: 9pt;
    color: #C0C0D0;
    padding: 6px;
}

/* ================================================================
   QScrollBar (vertical)
   ================================================================ */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #4A4A5E;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #5A5A6E;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}

/* horizontal */
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #4A4A5E;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #5A5A6E;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}

/* ================================================================
   QToolTip
   ================================================================ */
QToolTip {
    background-color: #252538;
    border: 1px solid #3A3A4E;
    color: #E0E0E0;
    padding: 4px 8px;
    border-radius: 4px;
}

/* ================================================================
   QStatusBar
   ================================================================ */
QStatusBar {
    background-color: #16162A;
    color: #B0B0C0;
    border-top: 1px solid #3A3A4E;
}
"""
