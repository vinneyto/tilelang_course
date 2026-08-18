import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("tilelang")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from course_utils import accelerator_device
from .task import build_affine_relu


def test_affine_relu_with_tail():
    device = accelerator_device()
    N, scale, bias = 1003, 1.75, -0.2
    x = torch.randn(N, device=device)
    out = torch.empty_like(x)
    kernel = build_affine_relu(N, scale, bias)
    kernel(x, out)
    torch.testing.assert_close(out, torch.relu(x * scale + bias))

