import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("tilelang")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from course_utils import accelerator_device
from task import build_row_bias_relu


def test_row_bias_relu():
    device = accelerator_device()
    M, N = 37, 53
    x = torch.randn(M, N, device=device)
    bias = torch.randn(N, device=device)
    out = torch.empty_like(x)
    build_row_bias_relu(M, N)(x, bias, out)
    torch.testing.assert_close(out, torch.relu(x + bias))

