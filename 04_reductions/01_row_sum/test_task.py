import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("tilelang")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from course_utils import accelerator_device
from task import build_row_sum


def test_row_sum():
    device = accelerator_device()
    M, N = 32, 256
    x = torch.randn(M, N, device=device)
    out = torch.empty(M, device=device)
    build_row_sum(M, N)(x, out)
    torch.testing.assert_close(out, x.sum(dim=1), rtol=1e-4, atol=1e-4)

