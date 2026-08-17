import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("tilelang")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from course_utils import accelerator_device
from task import build_vector_add


@pytest.mark.parametrize("N", [1024, 1003])
def test_vector_add(N: int):
    device = accelerator_device()
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    out = torch.empty_like(a)
    kernel = build_vector_add(N)
    kernel(a, b, out)
    torch.testing.assert_close(out, a + b)

