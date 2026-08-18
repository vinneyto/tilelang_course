import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("tilelang")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from course_utils import accelerator_device
from .task import build_tiled_square


def test_tiled_square():
    device = accelerator_device()
    N = 2048
    x = torch.randn(N, device=device)
    out = torch.empty_like(x)
    build_tiled_square(N)(x, out)
    torch.testing.assert_close(out, x.square())

