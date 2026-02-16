# Subtitle Extractor - Audio/Video Subtitle Extraction Tool

An AI-powered audio/video subtitle generation tool that supports multiple ASR (Automatic Speech Recognition) engines, provides a graphical user interface, and can extract text from video or audio files to generate SRT and CSV format subtitle files.

[中文版 README](./README.md) | [English README](./README_en.md)

## Features

- 🎬 Supports multiple audio/video formats (MP4, AVI, MKV, MOV, MP3, WAV, etc.)
- 🤖 Supports multiple ASR engines (Whisper, Qwen, etc.)
- 🔊 Voice Activity Detection (VAD) and audio enhancement
- 👥 Optional speaker diarization (CAM++ model)
- 🌍 Optional subtitle translation (supports multilingual translation)
- 📄 Outputs multiple formats: SRT subtitle files and CSV data tables
- 🎨 Intuitive graphical user interface (PyQt5)
- ⚙️ Rich parameter configuration options
- 🛡️ Supports processing interruption and partial result saving
- 💻 Supports CPU/GPU acceleration

## Technical Architecture

The project adopts a modular design and mainly includes the following components:

- **ASR Module**: Automatic speech recognition, supporting Whisper and other engines
- **VAD Processor**: Voice activity detection, accurately locating voice segments
- **Audio Processor**: Audio extraction and enhancement, improving recognition quality
- **Speaker Diarizer**: Distinguishes different speakers (optional feature)
- **Translation Module**: Subtitle translation, supporting multilingual translation
- **Subtitle Generator**: Generates final subtitle files
- **GUI Interface**: PyQt5 graphical interface for easy operation

## Installation Guide

### Requirements

- Python 3.12+
- pip package manager

### Installation Steps

1. Clone the project:
   ```bash
   git clone git@github.com:zong307/SubtitleExtractor.git
   cd SubtitleExtractor
   ```

2. Create virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   python app.py
   ```

## Usage

1. After running the program, click the "Browse" button to select the audio/video file to process
2. Configure ASR model type, size, and running device
3. Set processing parameters (language, VAD threshold, etc.)
4. Select output file path (CSV and SRT formats)
5. Click the "Start Processing" button to extract subtitles
6. View processing progress and log output

## Parameter Description

- **Model Type**: Select ASR engine (e.g., Whisper)
- **Model Size**: Select model size (affects accuracy and speed)
- **Device**: Select CPU or GPU for processing
- **Target Language**: Specify the primary language of the audio
- **VAD Threshold**: Sensitivity of voice activity detection
- **Silence End Delay**: Time to wait after detecting silence
- **Segment Expansion**: Time expansion before and after voice segments
- **Subtitle Character Limit**: Maximum character count per subtitle

## Docker Deployment (Optional)

> In development

## Development Guide

The project follows modular design principles with the main directory structure as follows:

```
src/
├── asr/           # ASR engine implementation
├── config/        # Configuration management
├── core/          # Core processing flow
├── models/        # Data model definitions
├── ui/            # User interface
├── utils/         # Utility functions
└── main.py        # Application entry point
```

## Contributing

Welcome to submit Issues and Pull Requests to improve this project.

## License

Please refer to the LICENSE file for details.

## Open Source Project References

We thank the following open source projects that this project depends on:

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) - Alibaba Tongyi Qwen ASR model
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - Graphical interface framework
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Deep learning model library
- [PyTorch](https://pytorch.org/) - Machine learning framework
- [FFmpeg](https://ffmpeg.org/) - Audio/video processing tool
- [Google TranslateGemma](https://github.com/google/gemma) - Google Gemma series models (for translation functionality)
- [FunASR](https://github.com/modelscope/funasr) - Alibaba open-source ASR toolkit
- [Loguru](https://github.com/Delgan/loguru) - Logging library
- [Scipy](https://github.com/scipy/scipy) - Scientific computing library
- [Numpy](https://numpy.org/) - Numerical computation foundation library
- [FFmpeg-Python](https://github.com/kkroening/ffmpeg-python) - FFmpeg Python interface
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library