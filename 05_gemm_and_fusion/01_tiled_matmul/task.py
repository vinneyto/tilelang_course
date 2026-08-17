import tilelang
import tilelang.language as T


def build_matmul(
    M: int,
    N: int,
    K: int,
    block_M: int = 32,
    block_N: int = 32,
    block_K: int = 16,
):
    """Скомпилировать Metal/CUDA-compatible GEMM с float32 shared accumulator."""
    raise NotImplementedError
