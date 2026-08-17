import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("tilelang")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from course_utils import accelerator_device
from task import build_matmul_relu


def test_matmul_relu():
    device = accelerator_device()
    M = N = K = 128
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    out = torch.empty(M, N, device=device, dtype=torch.float16)
    build_matmul_relu(M, N, K)(a, b, out)
    torch.testing.assert_close(out, torch.relu(a @ b), rtol=1e-2, atol=1e-2)

