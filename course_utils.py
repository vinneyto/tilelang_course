"""Общие утилиты тестов курса."""

from __future__ import annotations

import torch


def accelerator_device() -> torch.device:
    """Вернуть доступный CUDA/MPS accelerator или пропустить pytest-тест."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    import pytest

    pytest.skip("Для выполнения ядра нужен CUDA или поддерживаемый Metal/MPS backend")

