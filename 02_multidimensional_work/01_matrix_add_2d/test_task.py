import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("tilelang")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from course_utils import accelerator_device
from task import build_matrix_add


@pytest.mark.parametrize("shape", [(64, 96), (37, 53)])
def test_matrix_add(shape: tuple[int, int]):
    device = accelerator_device()
    M, N = shape
    a = torch.randn(M, N, device=device)
    b = torch.randn(M, N, device=device)
    out = torch.empty_like(a)
    build_matrix_add(M, N)(a, b, out)
    torch.testing.assert_close(out, a + b)

