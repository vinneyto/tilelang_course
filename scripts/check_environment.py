"""Короткая диагностика окружения TileLang."""

import platform

import tilelang
import torch

print(f"OS: {platform.platform()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"TileLang: {tilelang.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if platform.system() == "Darwin" and not torch.backends.mps.is_available():
    raise SystemExit("MPS недоступен: проверьте arm64 Python, PyTorch и версию macOS")
