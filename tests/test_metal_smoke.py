"""Решённый smoke-test установки TileLang на Apple Silicon."""

import pytest
import torch

tilelang = pytest.importorskip("tilelang")
import tilelang.language as T


@tilelang.jit
def vector_add(N: int, block: int = 256):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                global_i = bx * block + i
                C[global_i] = A[global_i] + B[global_i]

    return kernel


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS недоступен")
def test_tilelang_executes_on_metal():
    N = 1024
    a = torch.randn(N, device="mps", dtype=torch.float32)
    b = torch.randn(N, device="mps", dtype=torch.float32)
    out = torch.empty_like(a)

    vector_add(N)(a, b, out)

    torch.mps.synchronize()
    torch.testing.assert_close(out, a + b)

