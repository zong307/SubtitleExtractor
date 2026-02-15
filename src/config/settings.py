"""Settings manager: JSON-based configuration with auto-save/load."""

from __future__ import annotations

import copy
import json
import os
import platform
import threading
from pathlib import Path
from typing import Any, Optional

from src.config.defaults import DEFAULTS


def _get_config_dir() -> Path:
    """Return the platform-appropriate configuration directory."""
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
        return Path(base) / "subtitle_extractor"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "subtitle_extractor"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        return Path(xdg) / "subtitle_extractor"


class SettingsManager:
    """Thread-safe JSON-based settings manager with dot-notation access."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path:
            self._path = Path(config_path)
        else:
            self._path = _get_config_dir() / "settings.json"
        self._lock = threading.Lock()
        self._data: dict = {}
        self.load()

    def load(self) -> dict:
        """Load settings from disk, merging with defaults."""
        with self._lock:
            self._data = copy.deepcopy(DEFAULTS)
            if self._path.exists():
                try:
                    with open(self._path, "r", encoding="utf-8") as f:
                        user_data = json.load(f)
                    self._deep_merge(self._data, user_data)
                except (json.JSONDecodeError, OSError):
                    pass  # corrupted file -> use defaults
            return copy.deepcopy(self._data)

    def save(self) -> None:
        """Persist current settings to disk."""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a value using dot-notation path, e.g. 'vad.threshold'."""
        with self._lock:
            keys = key_path.split(".")
            node = self._data
            for key in keys:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    return default
            return node

    def set(self, key_path: str, value: Any) -> None:
        """Set a value using dot-notation path, then auto-save."""
        with self._lock:
            keys = key_path.split(".")
            node = self._data
            for key in keys[:-1]:
                if key not in node or not isinstance(node[key], dict):
                    node[key] = {}
                node = node[key]
            node[keys[-1]] = value
        self.save()

    def get_all(self) -> dict:
        """Return a deep copy of all settings."""
        with self._lock:
            return copy.deepcopy(self._data)

    def reset(self) -> None:
        """Reset all settings to defaults and save."""
        with self._lock:
            self._data = copy.deepcopy(DEFAULTS)
        self.save()

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        """Recursively merge override into base (in-place)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                SettingsManager._deep_merge(base[key], value)
            else:
                base[key] = value
