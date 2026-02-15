"""CPU/GPU device detection utilities."""

from __future__ import annotations

from loguru import logger


def get_available_devices() -> list[str]:
    """Return a list of available compute devices."""
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
    except ImportError:
        pass
    return devices


def get_device(preferred: str = "cpu") -> str:
    """Validate and return a usable device, falling back to CPU."""
    available = get_available_devices()
    if preferred in available:
        return preferred
    if preferred.startswith("cuda") and preferred not in available:
        logger.warning(
            f"Requested device '{preferred}' is not available. Falling back to CPU."
        )
    return "cpu"


def get_device_display_info() -> str:
    """Return a short string describing the compute environment for the UI status bar."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"GPU: {name}"
    except ImportError:
        pass
    return "CPU only"
