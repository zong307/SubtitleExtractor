"""Default configuration values for the subtitle extractor."""

DEFAULTS: dict = {
    "asr": {
        "type": "whisper",
        "model_size": "turbo",
        "device": "cpu",
        "language": "zh",
    },
    "vad": {
        "threshold": 0.5,
        "min_silence_duration_ms": 300,
        "min_speech_duration_ms": 250,
        "speech_pad_ms": 30,
        "max_speech_duration_s": 30.0,
    },
    "audio": {
        "padding_seconds": 2.0,
        "enable_noise_reduce": True,
        "enable_bandpass": True,
    },
    "subtitle": {
        "max_chars_per_subtitle": 30,
    },
    "diarization": {
        "enabled": False,
    },
    "paths": {
        "model_dir": "",  # empty = use each library's default cache
    },
    "translation": {
        "enabled": False,
        "target_language": "en",
        "model_size": "4b",
    },
    "huggingface": {
        "endpoint": "",  # Default: use official, "" means official
    },
}

# Languages supported for UI display (code -> display name)
LANGUAGES: dict[str, str] = {
    "zh": "中文 (zh)",
    "en": "英语 (en)",
    "ja": "日语 (ja)",
    "ko": "朝鲜语 (ko)",
    "fr": "法语 (fr)",
    "de": "德语 (de)",
    "es": "西班牙语 (es)",
    "ru": "俄语 (ru)",
    "ar": "阿拉伯语 (ar)",
    "auto": "自动检测",
}