"""Main application window – PyQt5 UI for the subtitle extractor."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from loguru import logger
from PyQt5.QtCore import Qt, pyqtSlot, QUrl
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices
from PyQt5.QtSvg import QSvgWidget  # Import SVG support
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.asr import ASRRegistry
from src.config.defaults import LANGUAGES
from src.config.settings import SettingsManager
from src.ui.log_handler import QtLogHandler
from src.ui.style import get_dark_stylesheet
from src.ui.worker import PipelineWorker
from src.utils.device import get_available_devices, get_device_display_info
from src.utils.logger import add_ui_sink

# Color map for log levels
_LOG_COLORS = {
    "DEBUG": "#6B7280",
    "INFO": "#3B82F6",
    "SUCCESS": "#10B981",
    "WARNING": "#F59E0B",
    "ERROR": "#EF4444",
    "CRITICAL": "#DC2626",
}


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Subtitle Extractor - 音视频字幕提取工具")
        self.setMinimumSize(960, 720)
        self.resize(1060, 800)

        self._settings = SettingsManager()
        self._worker: Optional[PipelineWorker] = None

        # Log handler (bridges loguru -> UI)
        self._log_handler = QtLogHandler()
        self._log_handler.log_received.connect(self._on_log_message)
        self._log_sink_id = add_ui_sink(self._log_handler.write, log_level="DEBUG")

        self._build_ui()
        self._load_settings()
        self._connect_signals()

        self.setStyleSheet(get_dark_stylesheet())

        logger.info("应用程序已启动")

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        # Create menu bar
        menubar = self.menuBar()
        options_menu = menubar.addMenu('选项')
        
        # Add HuggingFace endpoint submenu
        hf_endpoint_action = options_menu.addAction('HuggingFace endpoint')
        hf_endpoint_action.triggered.connect(self._show_hf_endpoint_dialog)
        
        # Scroll area for the whole content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        self._main_layout = QVBoxLayout(content)
        self._main_layout.setSpacing(8)
        self._main_layout.setContentsMargins(12, 8, 12, 8)

        self._build_input_section()
        self._build_model_section()
        self._build_params_section()
        self._build_output_section()
        self._build_progress_section()
        self._build_log_section()

        scroll.setWidget(content)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._device_label = QLabel(get_device_display_info())
        self._version_label = QLabel("v1.0.0")
        self._status_bar.addPermanentWidget(self._version_label)
        self._status_bar.addPermanentWidget(self._device_label)

    # -- Input file section ------------------------------------------------

    def _build_input_section(self) -> None:
        group = QGroupBox("输入文件")
        layout = QHBoxLayout()
        self._input_path = QLineEdit()
        self._input_path.setReadOnly(True)
        self._input_path.setPlaceholderText("选择一个视频或音频文件...")
        self._browse_input_btn = QPushButton("浏览...")
        layout.addWidget(self._input_path, 1)
        layout.addWidget(self._browse_input_btn)
        group.setLayout(layout)
        self._main_layout.addWidget(group)

    # -- Model / diarization section ---------------------------------------

    def _build_model_section(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)

        # ASR model config
        asr_group = QGroupBox("ASR 模型配置")
        asr_layout = QGridLayout()

        asr_layout.addWidget(QLabel("模型类型:"), 0, 0)
        self._asr_type_combo = QComboBox()
        for t in ASRRegistry.list_types():
            self._asr_type_combo.addItem(ASRRegistry.get_display_name(t), t)
        asr_layout.addWidget(self._asr_type_combo, 0, 1)

        asr_layout.addWidget(QLabel("模型规格:"), 1, 0)
        self._model_size_combo = QComboBox()
        asr_layout.addWidget(self._model_size_combo, 1, 1)

        asr_layout.addWidget(QLabel("计算设备:"), 2, 0)
        self._device_combo = QComboBox()
        for dev in get_available_devices():
            self._device_combo.addItem(dev)
        asr_layout.addWidget(self._device_combo, 2, 1)

        asr_group.setLayout(asr_layout)
        row.addWidget(asr_group, 1)

        # Speaker diarization
        diar_group = QGroupBox("可选功能")
        diar_layout = QVBoxLayout()
        self._enable_diarization = QCheckBox("启用说话人分离 (CAM++)")
        self._enable_diarization.setToolTip(
            "使用 CAM++ 模型自动识别不同说话人，并在字幕中标注说话人身份"
        )
        diar_layout.addWidget(self._enable_diarization)
        
        # Translation
        self._enable_translation = QCheckBox("启用字幕翻译")
        self._enable_translation.setToolTip(
            "使用 translategemma 模型翻译字幕"
        )
        diar_layout.addWidget(self._enable_translation)
        
        # Translation options container (will be shown/hidden based on checkbox)
        self._translation_options_widget = QWidget()
        translation_options_layout = QVBoxLayout()
        translation_options_layout.setContentsMargins(0, 0, 0, 0)
        translation_options_layout.setSpacing(4)
        
        # Model size row
        model_size_row = QHBoxLayout()
        model_size_row.addWidget(QLabel("模型尺寸:"))
        self._translation_model_size = QComboBox()
        self._translation_model_size.addItems(["4b", "12b"])
        self._translation_model_size.setCurrentText("4b")
        self._translation_model_size.setToolTip("选择翻译模型规模")
        model_size_row.addWidget(self._translation_model_size)
        model_size_row.addStretch()
        translation_options_layout.addLayout(model_size_row)
        
        # Source language row
        source_lang_row = QHBoxLayout()
        source_lang_row.addWidget(QLabel("原文语言:"))
        self._translation_source_lang = QComboBox()
        self._translation_source_lang.addItems([
            "英语 (en)", "中文 (zh)", "日语 (ja)", "朝鲜语 (ko)", "法语 (fr)",
            "德语 (de)", "西班牙语 (es)", "俄语 (ru)", "阿拉伯语 (ar)"
        ])
        self._translation_source_lang.setCurrentText("中文 (zh)")
        self._translation_source_lang.setToolTip("选择翻译源语言")
        source_lang_row.addWidget(self._translation_source_lang)
        source_lang_row.addStretch()
        translation_options_layout.addLayout(source_lang_row)
        
        # Target language row
        target_lang_row = QHBoxLayout()
        target_lang_row.addWidget(QLabel("译文语言:"))
        self._translation_target_lang = QComboBox()
        self._translation_target_lang.addItems([
            "英语 (en)", "中文 (zh)", "日语 (ja)", "朝鲜语 (ko)", "法语 (fr)",
            "德语 (de)", "西班牙语 (es)", "俄语 (ru)", "阿拉伯语 (ar)"
        ])
        self._translation_target_lang.setCurrentText("英语 (en)")
        self._translation_target_lang.setToolTip("选择翻译目标语言")
        target_lang_row.addWidget(self._translation_target_lang)
        target_lang_row.addStretch()
        translation_options_layout.addLayout(target_lang_row)
        
        self._translation_options_widget.setLayout(translation_options_layout)
        self._translation_options_widget.setVisible(False)  # Initially hidden
        diar_layout.addWidget(self._translation_options_widget)
        
        diar_layout.addStretch()
        diar_group.setLayout(diar_layout)
        row.addWidget(diar_group, 1)

        wrapper = QWidget()
        wrapper.setLayout(row)
        self._main_layout.addWidget(wrapper)

        # Populate model sizes for the initially selected ASR type
        self._update_model_sizes()

    # -- Parameters section ------------------------------------------------

    def _build_params_section(self) -> None:
        group = QGroupBox("处理参数")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)

        # Row 0
        grid.addWidget(QLabel("目标语言:"), 0, 0)
        self._language_combo = QComboBox()
        for code, display in LANGUAGES.items():
            self._language_combo.addItem(display, code)
        grid.addWidget(self._language_combo, 0, 1)

        grid.addWidget(QLabel("VAD 阈值:"), 0, 2)
        self._vad_threshold = QDoubleSpinBox()
        self._vad_threshold.setRange(0.01, 1.0)
        self._vad_threshold.setSingleStep(0.05)
        self._vad_threshold.setDecimals(2)
        self._vad_threshold.setToolTip("语音活动检测的置信度阈值，越高越严格")
        grid.addWidget(self._vad_threshold, 0, 3)

        # Row 1
        grid.addWidget(QLabel("静音结束延迟 (ms):"), 1, 0)
        self._silence_delay = QSpinBox()
        self._silence_delay.setRange(50, 5000)
        self._silence_delay.setSingleStep(50)
        self._silence_delay.setToolTip("检测到静音后等待多少毫秒才判定语音结束")
        grid.addWidget(self._silence_delay, 1, 1)

        grid.addWidget(QLabel("片段前后扩充 (秒):"), 1, 2)
        self._padding_spin = QDoubleSpinBox()
        self._padding_spin.setRange(0.0, 10.0)
        self._padding_spin.setSingleStep(0.5)
        self._padding_spin.setDecimals(1)
        self._padding_spin.setToolTip("对 VAD 检测到的每个语音片段前后各扩展的时长")
        grid.addWidget(self._padding_spin, 1, 3)

        # Row 2
        grid.addWidget(QLabel("单条字幕上限 (字):"), 2, 0)
        self._max_chars = QSpinBox()
        self._max_chars.setRange(10, 200)
        self._max_chars.setSingleStep(5)
        self._max_chars.setToolTip("单条字幕的最大字符数，超出会自动拆分")
        grid.addWidget(self._max_chars, 2, 1)

        grid.addWidget(QLabel("语音片段上限 (秒):"), 2, 2)
        self._max_speech_duration = QDoubleSpinBox()
        self._max_speech_duration.setRange(5.0, 300.0)
        self._max_speech_duration.setSingleStep(5.0)
        self._max_speech_duration.setDecimals(1)
        self._max_speech_duration.setToolTip(
            "单个语音片段的最大时长，超长片段将被强制切分后再送入 ASR"
        )
        grid.addWidget(self._max_speech_duration, 2, 3)

        # Row 3
        grid.addWidget(QLabel("模型存储路径:"), 3, 0)
        model_path_layout = QHBoxLayout()
        self._model_dir_input = QLineEdit()
        self._model_dir_input.setPlaceholderText("留空使用默认缓存路径")
        self._browse_model_dir_btn = QPushButton("更改...")
        model_path_layout.addWidget(self._model_dir_input, 1)
        model_path_layout.addWidget(self._browse_model_dir_btn)
        grid.addLayout(model_path_layout, 3, 1, 1, 3)

        # Row 4: reset button
        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton("重置为默认值")
        btn_row.addStretch()
        btn_row.addWidget(self._reset_btn)
        grid.addLayout(btn_row, 4, 0, 1, 4)

        group.setLayout(grid)
        self._main_layout.addWidget(group)

    # -- Output section ----------------------------------------------------

    def _build_output_section(self) -> None:
        group = QGroupBox("输出设置")
        grid = QGridLayout()

        grid.addWidget(QLabel("CSV 输出路径:"), 0, 0)
        self._csv_path = QLineEdit()
        self._csv_path.setPlaceholderText("选择 CSV 保存路径...")
        self._browse_csv_btn = QPushButton("浏览...")
        grid.addWidget(self._csv_path, 0, 1)
        grid.addWidget(self._browse_csv_btn, 0, 2)

        grid.addWidget(QLabel("SRT 输出路径:"), 1, 0)
        self._srt_path = QLineEdit()
        self._srt_path.setPlaceholderText("选择 SRT 保存路径...")
        self._browse_srt_btn = QPushButton("浏览...")
        grid.addWidget(self._srt_path, 1, 1)
        grid.addWidget(self._browse_srt_btn, 1, 2)

        group.setLayout(grid)
        self._main_layout.addWidget(group)

    # -- Progress section --------------------------------------------------

    def _build_progress_section(self) -> None:
        group = QGroupBox("处理进度")
        layout = QVBoxLayout()

        self._status_label = QLabel("就绪")
        self._status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet("color: #B0B0C0;")
        layout.addWidget(self._detail_label)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("开始处理")
        self._start_btn.setObjectName("start_btn")
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        group.setLayout(layout)
        self._main_layout.addWidget(group)

    # -- Log section -------------------------------------------------------

    def _build_log_section(self) -> None:
        group = QGroupBox("日志输出")
        layout = QVBoxLayout()

        self._log_viewer = QTextEdit()
        self._log_viewer.setObjectName("log_viewer")
        self._log_viewer.setReadOnly(True)
        self._log_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._log_viewer.setMinimumHeight(120)
        layout.addWidget(self._log_viewer, 1)

        btn_row = QHBoxLayout()
        self._clear_log_btn = QPushButton("清空日志")
        self._export_log_btn = QPushButton("导出日志")
        btn_row.addStretch()
        btn_row.addWidget(self._clear_log_btn)
        btn_row.addWidget(self._export_log_btn)
        layout.addLayout(btn_row)

        group.setLayout(layout)
        self._main_layout.addWidget(group, 1)
        
        # Add GitHub link and author info
        github_row = QHBoxLayout()
        github_row.addStretch()  # Left padding
        
        # Create a clickable SVG widget for GitHub logo
        github_svg_widget = QSvgWidget()
        github_svg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui', 'pic', 'GitHub_Lockup_White.svg')
        github_svg_widget.load(github_svg_path)
        github_svg_widget.setFixedSize(60, 14)  # Adjust size to fit the GitHub logo appropriately
        github_svg_widget.setStyleSheet("background: transparent;")  # Make background transparent
        
        # Create a clickable label that will handle the click event
        github_clickable_label = QLabel()
        github_clickable_label.setPixmap(github_svg_widget.grab())  # Grab the rendered SVG as pixmap
        github_clickable_label.setCursor(Qt.PointingHandCursor)  # Set cursor to hand pointer
        github_clickable_label.mousePressEvent = lambda event: QDesktopServices.openUrl(QUrl("https://github.com/zong307/SubtitleExtractor"))
        
        # Add the clickable GitHub logo to the layout
        github_row.addWidget(github_clickable_label)
        
        # Add author info
        author_label = QLabel("© 2026 zong307 | MIT License")
        author_label.setAlignment(Qt.AlignCenter)
        github_row.addWidget(author_label)
        
        github_row.addStretch()  # Right padding
        
        # Add the row to the main layout
        self._main_layout.addLayout(github_row)

    # ==================================================================
    # Signal / Slot Wiring
    # ==================================================================

    def _connect_signals(self) -> None:
        # Buttons
        self._browse_input_btn.clicked.connect(self._browse_input_file)
        self._browse_csv_btn.clicked.connect(lambda: self._browse_save_file("csv"))
        self._browse_srt_btn.clicked.connect(lambda: self._browse_save_file("srt"))
        self._browse_model_dir_btn.clicked.connect(self._browse_model_dir)
        self._start_btn.clicked.connect(self._start_processing)
        self._stop_btn.clicked.connect(self._stop_processing)
        self._clear_log_btn.clicked.connect(self._log_viewer.clear)
        self._export_log_btn.clicked.connect(self._export_log)
        self._reset_btn.clicked.connect(self._reset_settings)

        # ASR type change -> update model size options
        self._asr_type_combo.currentIndexChanged.connect(self._update_model_sizes)

        # Auto-save on any change
        self._asr_type_combo.currentIndexChanged.connect(self._auto_save)
        self._model_size_combo.currentIndexChanged.connect(self._auto_save)
        self._device_combo.currentIndexChanged.connect(self._auto_save)
        self._enable_diarization.stateChanged.connect(self._auto_save)
        self._enable_translation.stateChanged.connect(self._auto_save)
        self._translation_target_lang.currentTextChanged.connect(self._auto_save)
        self._translation_model_size.currentTextChanged.connect(self._auto_save)
        self._translation_source_lang.currentTextChanged.connect(self._auto_save)
        self._language_combo.currentIndexChanged.connect(self._auto_save)
        # Connect ASR target language change to sync with translation source language
        self._language_combo.currentIndexChanged.connect(self._sync_asr_language_to_translation)
        self._vad_threshold.valueChanged.connect(self._auto_save)
        self._silence_delay.valueChanged.connect(self._auto_save)
        self._padding_spin.valueChanged.connect(self._auto_save)
        self._max_chars.valueChanged.connect(self._auto_save)
        self._max_speech_duration.valueChanged.connect(self._auto_save)
        self._model_dir_input.editingFinished.connect(self._auto_save)
        
        # Enable/disable translation UI elements based on checkbox
        self._enable_translation.stateChanged.connect(self._toggle_translation_options)

    # ==================================================================
    # Slots
    # ==================================================================

    def _browse_input_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频/视频文件",
            "",
            "媒体文件 (*.mp4 *.avi *.mkv *.mov *.flv *.webm *.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma);;所有文件 (*)",
        )
        if path:
            self._input_path.setText(path)
            # Always update output paths to match the new input file
            base = os.path.splitext(path)[0]
            self._csv_path.setText(base + ".csv")
            self._srt_path.setText(base + ".srt")

    def _browse_save_file(self, fmt: str) -> None:
        if fmt == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "保存 CSV 文件", "", "CSV 文件 (*.csv)"
            )
            if path:
                self._csv_path.setText(path)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "保存 SRT 字幕文件", "", "SRT 字幕 (*.srt)"
            )
            if path:
                self._srt_path.setText(path)

    def _browse_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择模型存储目录")
        if path:
            self._model_dir_input.setText(path)
            self._auto_save()

    def _update_model_sizes(self) -> None:
        asr_type = self._asr_type_combo.currentData()
        if not asr_type:
            return
        self._model_size_combo.blockSignals(True)
        self._model_size_combo.clear()
        for size in ASRRegistry.get_model_sizes(asr_type):
            self._model_size_combo.addItem(size)
        self._model_size_combo.blockSignals(False)

    def _start_processing(self) -> None:
        # Validate
        input_path = self._input_path.text().strip()
        csv_path = self._csv_path.text().strip()
        srt_path = self._srt_path.text().strip()

        if not input_path:
            QMessageBox.warning(self, "提示", "请先选择输入文件。")
            return
        if not os.path.isfile(input_path):
            QMessageBox.warning(self, "提示", f"输入文件不存在:\n{input_path}")
            return
        if not csv_path or not srt_path:
            QMessageBox.warning(self, "提示", "请设置 CSV 和 SRT 输出路径。")
            return

        # Build config from UI
        config = self._collect_config()

        # Disable UI
        self._set_ui_enabled(False)
        self._progress_bar.setValue(0)
        self._status_label.setText("正在初始化...")
        self._detail_label.setText("")

        # Create and start worker
        self._worker = PipelineWorker(config, input_path, csv_path, srt_path)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.finished_cancelled.connect(self._on_finished_cancelled)
        self._worker.start()

    def _stop_processing(self) -> None:
        reply = QMessageBox.question(
            self,
            "确认停止",
            "确定要停止处理吗？已处理的数据将会保存。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes and self._worker is not None:
            self._worker.cancel()
            self._stop_btn.setEnabled(False)
            self._status_label.setText("正在停止...")

    @pyqtSlot(str, float, str)
    def _on_progress(self, step: str, pct: float, msg: str) -> None:
        self._progress_bar.setValue(int(pct))
        self._detail_label.setText(msg)
        if step == "done":
            self._status_label.setText("完成")
            self._status_label.setStyleSheet("font-weight: bold; color: #10B981;")
        else:
            self._status_label.setText("处理中...")

    @pyqtSlot(str)
    def _on_finished_ok(self, msg: str) -> None:
        self._set_ui_enabled(True)
        self._status_label.setText("完成")
        self._status_label.setStyleSheet("font-weight: bold; color: #10B981;")
        QMessageBox.information(self, "处理完成", msg)

    @pyqtSlot(str)
    def _on_finished_error(self, msg: str) -> None:
        self._set_ui_enabled(True)
        self._status_label.setText("失败")
        self._status_label.setStyleSheet("font-weight: bold; color: #EF4444;")
        QMessageBox.critical(self, "处理失败", f"发生错误:\n{msg}")

    @pyqtSlot(str)
    def _on_finished_cancelled(self, msg: str) -> None:
        self._set_ui_enabled(True)
        self._status_label.setText("已取消")
        self._status_label.setStyleSheet("font-weight: bold; color: #F59E0B;")
        QMessageBox.information(self, "已取消", msg)

    @pyqtSlot(str, str)
    def _on_log_message(self, level: str, text: str) -> None:
        color = _LOG_COLORS.get(level, "#E0E0E0")
        escaped = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        html = f'<span style="color:{color};">{escaped}</span>'
        self._log_viewer.append(html)
        # Auto-scroll to bottom
        sb = self._log_viewer.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _export_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", "", "文本文件 (*.txt);;所有文件 (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._log_viewer.toPlainText())
            logger.info(f"日志已导出至: {path}")

    def _toggle_translation_options(self, state):
        """Show or hide translation options based on checkbox state."""
        enabled = bool(state)
        self._translation_options_widget.setVisible(enabled)

    def _sync_asr_language_to_translation(self):
        """Sync the ASR target language to translation source language when changed."""
        # Get the current ASR target language
        asr_lang_code = self._language_combo.currentData()
        
        # Convert the language code to display text
        lang_display_map = {
            "en": "英语 (en)",
            "zh": "中文 (zh)",
            "ja": "日语 (ja)",
            "ko": "朝鲜语 (ko)",
            "fr": "法语 (fr)",
            "de": "德语 (de)",
            "es": "西班牙语 (es)",
            "ru": "俄语 (ru)",
            "ar": "阿拉伯语 (ar)"
        }
        
        asr_lang_display = lang_display_map.get(asr_lang_code, "中文 (zh)")
        
        # Find and set the corresponding index in the translation source language combo
        source_idx = self._translation_source_lang.findText(asr_lang_display)
        if source_idx >= 0:
            self._translation_source_lang.setCurrentIndex(source_idx)

    def _reset_settings(self) -> None:
        reply = QMessageBox.question(
            self,
            "重置设置",
            "确定将所有参数重置为默认值吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._settings.reset()
            self._load_settings()
            logger.info("设置已重置为默认值")

    # ==================================================================
    # Settings persistence
    # ==================================================================

    def _load_settings(self) -> None:
        """Populate UI widgets from saved settings."""
        s = self._settings

        # ASR
        asr_type = s.get("asr.type", "whisper")
        idx = self._asr_type_combo.findData(asr_type)
        if idx >= 0:
            self._asr_type_combo.setCurrentIndex(idx)
        self._update_model_sizes()
        model_size = s.get("asr.model_size", "turbo")
        idx = self._model_size_combo.findText(model_size)
        if idx >= 0:
            self._model_size_combo.setCurrentIndex(idx)

        device = s.get("asr.device", "cpu")
        idx = self._device_combo.findText(device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

        # Language
        lang = s.get("asr.language", "zh")
        idx = self._language_combo.findData(lang)
        if idx >= 0:
            self._language_combo.setCurrentIndex(idx)

        # VAD
        self._vad_threshold.setValue(s.get("vad.threshold", 0.5))
        self._silence_delay.setValue(s.get("vad.min_silence_duration_ms", 300))
        self._max_speech_duration.setValue(s.get("vad.max_speech_duration_s", 30.0))

        # Audio
        self._padding_spin.setValue(s.get("audio.padding_seconds", 2.0))

        # Subtitle
        self._max_chars.setValue(s.get("subtitle.max_chars_per_subtitle", 30))

        # Diarization
        self._enable_diarization.setChecked(s.get("diarization.enabled", False))
        
        # Translation
        self._enable_translation.setChecked(s.get("translation.enabled", False))
        
        # Set translation target language
        target_lang = s.get("translation.target_language", "en")
        lang_display_map = {
            "en": "英语 (en)",
            "zh": "中文 (zh)",
            "ja": "日语 (ja)",
            "ko": "朝鲜语 (ko)",
            "fr": "法语 (fr)",
            "de": "德语 (de)",
            "es": "西班牙语 (es)",
            "ru": "俄语 (ru)",
            "ar": "阿拉伯语 (ar)"
        }
        target_lang_display = lang_display_map.get(target_lang, "英语 (en)")
        target_idx = self._translation_target_lang.findText(target_lang_display)
        if target_idx >= 0:
            self._translation_target_lang.setCurrentIndex(target_idx)
        
        # Set translation source language
        source_lang = s.get("translation.source_language", "zh")
        source_lang_display = lang_display_map.get(source_lang, "中文 (zh)")
        source_idx = self._translation_source_lang.findText(source_lang_display)
        if source_idx >= 0:
            self._translation_source_lang.setCurrentIndex(source_idx)
        
        # Set translation model size
        model_size = s.get("translation.model_size", "4b")
        size_idx = self._translation_model_size.findText(model_size)
        if size_idx >= 0:
            self._translation_model_size.setCurrentIndex(size_idx)
        
        # Update UI state based on translation setting
        self._toggle_translation_options(self._enable_translation.isChecked())
        
        # Paths
        self._model_dir_input.setText(s.get("paths.model_dir", ""))
        
        # HuggingFace endpoint - currently no UI element to load this setting to since it's in a dialog

    def _auto_save(self) -> None:
        """Persist current UI state to settings file."""
        s = self._settings
        s.set("asr.type", self._asr_type_combo.currentData() or "whisper")
        s.set("asr.model_size", self._model_size_combo.currentText())
        s.set("asr.device", self._device_combo.currentText())
        s.set("asr.language", self._language_combo.currentData() or "zh")
        s.set("vad.threshold", self._vad_threshold.value())
        s.set("vad.min_silence_duration_ms", self._silence_delay.value())
        s.set("vad.max_speech_duration_s", self._max_speech_duration.value())
        s.set("audio.padding_seconds", self._padding_spin.value())
        s.set("subtitle.max_chars_per_subtitle", self._max_chars.value())
        s.set("diarization.enabled", self._enable_diarization.isChecked())
        s.set("translation.enabled", self._enable_translation.isChecked())
        
        # Map UI language selection to language code
        lang_text = self._translation_target_lang.currentText()
        lang_code = lang_text.split('(')[-1].replace(')', '')  # Extract code from "English (en)"
        s.set("translation.target_language", lang_code)
        
        # Map UI source language selection to language code
        source_lang_text = self._translation_source_lang.currentText()
        source_lang_code = source_lang_text.split('(')[-1].replace(')', '')  # Extract code from "English (en)"
        s.set("translation.source_language", source_lang_code)
        
        s.set("translation.model_size", self._translation_model_size.currentText())
        
        s.set("paths.model_dir", self._model_dir_input.text().strip())
        
        # HuggingFace endpoint is managed through the dialog, not auto-saved from UI elements

    def _collect_config(self) -> dict:
        """Build the pipeline config dict from current UI values."""
        # Get current settings as base
        config = self._settings.get_all()
        
        # Override with current UI values to ensure latest state is captured
        config['diarization']['enabled'] = self._enable_diarization.isChecked()
        config['translation']['enabled'] = self._enable_translation.isChecked()
        
        # Map UI language selection to language code
        lang_text = self._translation_target_lang.currentText()
        lang_code = lang_text.split('(')[-1].replace(')', '')  # Extract code from "English (en)"
        config['translation']['target_language'] = lang_code
        
        # Map UI source language selection to language code
        source_lang_text = self._translation_source_lang.currentText()
        source_lang_code = source_lang_text.split('(')[-1].replace(')', '')  # Extract code from "English (en)"
        config['translation']['source_language'] = source_lang_code
        
        config['translation']['model_size'] = self._translation_model_size.currentText()
        
        # Ensure huggingface endpoint is included in the config
        config['huggingface']['endpoint'] = self._settings.get("huggingface.endpoint", "")
        
        return config

    # ==================================================================
    # Helpers
    # ==================================================================

    def _set_ui_enabled(self, enabled: bool) -> None:
        """Toggle input controls and start/stop buttons."""
        self._start_btn.setEnabled(enabled)
        self._stop_btn.setEnabled(not enabled)
        self._browse_input_btn.setEnabled(enabled)
        self._browse_csv_btn.setEnabled(enabled)
        self._browse_srt_btn.setEnabled(enabled)
        self._asr_type_combo.setEnabled(enabled)
        self._model_size_combo.setEnabled(enabled)
        self._device_combo.setEnabled(enabled)
        self._enable_diarization.setEnabled(enabled)
        self._language_combo.setEnabled(enabled)
        self._vad_threshold.setEnabled(enabled)
        self._silence_delay.setEnabled(enabled)
        self._padding_spin.setEnabled(enabled)
        self._max_chars.setEnabled(enabled)
        self._max_speech_duration.setEnabled(enabled)
        self._model_dir_input.setEnabled(enabled)
        self._browse_model_dir_btn.setEnabled(enabled)
        self._reset_btn.setEnabled(enabled)
        
        # Enable translation-related widgets
        self._enable_translation.setEnabled(enabled)
        self._translation_model_size.setEnabled(enabled)
        self._translation_source_lang.setEnabled(enabled)
        self._translation_target_lang.setEnabled(enabled)
        
        if not enabled:
            self._status_label.setStyleSheet("font-weight: bold; color: #E0E0E0;")
        
        # 日志输出部分的按钮始终可用，所以不需要改变它们的状态
        # self._clear_log_btn 和 self._export_log_btn 始终保持启用状态

    def closeEvent(self, event) -> None:  # noqa: N802
        self._auto_save()
        # If worker is running, ask to stop
        if self._worker is not None and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "处理仍在运行中，确定退出吗？已处理的数据将会保存。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait(5000)
        event.accept()

    def _show_hf_endpoint_dialog(self) -> None:
        """Show dialog to configure HuggingFace endpoint."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QLineEdit, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("HuggingFace Endpoint 设置")
        dialog.resize(400, 120)
        
        layout = QVBoxLayout()
        
        # Label
        label = QLabel("请选择 HuggingFace 镜像地址:")
        layout.addWidget(label)
        
        # ComboBox for predefined options
        combo_layout = QHBoxLayout()
        combo_label = QLabel("预设选项:")
        self._hf_endpoint_combo = QComboBox()
        self._hf_endpoint_combo.addItems([
            "官网默认", 
            "大陆代理 (https://hf-mirror.com)",
            "自定义"
        ])
        
        # Load current setting to determine combo box selection
        current_endpoint = self._settings.get("huggingface.endpoint", "")
        if current_endpoint == "":
            self._hf_endpoint_combo.setCurrentIndex(0)  # 官网默认
        elif current_endpoint == "https://hf-mirror.com":
            self._hf_endpoint_combo.setCurrentIndex(1)  # 大陆代理
        else:
            self._hf_endpoint_combo.setCurrentIndex(2)  # 自定义
        
        combo_layout.addWidget(combo_label)
        combo_layout.addWidget(self._hf_endpoint_combo, 1)
        layout.addLayout(combo_layout)
        
        # Custom URL input (initially hidden)
        custom_layout = QHBoxLayout()
        custom_label = QLabel("自定义地址:")
        self._custom_endpoint_input = QLineEdit()
        self._custom_endpoint_input.setText(current_endpoint if current_endpoint not in ["", "https://hf-mirror.com"] else "")
        self._custom_endpoint_input.setPlaceholderText("请输入自定义 HuggingFace 镜像地址")
        
        custom_layout.addWidget(custom_label)
        custom_layout.addWidget(self._custom_endpoint_input, 1)
        
        # Show custom input only if current selection is custom
        self._custom_endpoint_input.setVisible(self._hf_endpoint_combo.currentIndex() == 2)
        
        layout.addLayout(custom_layout)
        
        # Connect combo box change signal to show/hide custom input
        self._hf_endpoint_combo.currentTextChanged.connect(
            lambda text: self._custom_endpoint_input.setVisible(text == "自定义")
        )
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # Save the selected endpoint
            selected_option = self._hf_endpoint_combo.currentText()
            if selected_option == "官网默认":
                endpoint = ""
            elif selected_option == "大陆代理 (https://hf-mirror.com)":
                endpoint = "https://hf-mirror.com"
            else:  # 自定义
                endpoint = self._custom_endpoint_input.text().strip()
                # If custom endpoint is not empty, add it to the combo options
                if endpoint and endpoint not in ["", "https://hf-mirror.com"]:
                    # In a real implementation, we might want to store custom endpoints
                    # but for this implementation we just save the endpoint value
                    pass
            
            self._settings.set("huggingface.endpoint", endpoint)
            logger.info(f"HuggingFace endpoint 已设置为: {endpoint or '官网默认'}")

    def _connect_signals(self) -> None:
        # Buttons
        self._browse_input_btn.clicked.connect(self._browse_input_file)
        self._browse_csv_btn.clicked.connect(lambda: self._browse_save_file("csv"))
        self._browse_srt_btn.clicked.connect(lambda: self._browse_save_file("srt"))
        self._browse_model_dir_btn.clicked.connect(self._browse_model_dir)
        self._start_btn.clicked.connect(self._start_processing)
        self._stop_btn.clicked.connect(self._stop_processing)
        self._clear_log_btn.clicked.connect(self._log_viewer.clear)
        self._export_log_btn.clicked.connect(self._export_log)
        self._reset_btn.clicked.connect(self._reset_settings)

        # ASR type change -> update model size options
        self._asr_type_combo.currentIndexChanged.connect(self._update_model_sizes)

        # Auto-save on any change
        self._asr_type_combo.currentIndexChanged.connect(self._auto_save)
        self._model_size_combo.currentIndexChanged.connect(self._auto_save)
        self._device_combo.currentIndexChanged.connect(self._auto_save)
        self._enable_diarization.stateChanged.connect(self._auto_save)
        self._enable_translation.stateChanged.connect(self._auto_save)
        self._translation_target_lang.currentTextChanged.connect(self._auto_save)
        self._translation_model_size.currentTextChanged.connect(self._auto_save)
        self._translation_source_lang.currentTextChanged.connect(self._auto_save)
        self._language_combo.currentIndexChanged.connect(self._auto_save)
        # Connect ASR target language change to sync with translation source language
        self._language_combo.currentIndexChanged.connect(self._sync_asr_language_to_translation)
        self._vad_threshold.valueChanged.connect(self._auto_save)
        self._silence_delay.valueChanged.connect(self._auto_save)
        self._padding_spin.valueChanged.connect(self._auto_save)
        self._max_chars.valueChanged.connect(self._auto_save)
        self._max_speech_duration.valueChanged.connect(self._auto_save)
        self._model_dir_input.editingFinished.connect(self._auto_save)
        
        # Enable/disable translation UI elements based on checkbox
        self._enable_translation.stateChanged.connect(self._toggle_translation_options)
