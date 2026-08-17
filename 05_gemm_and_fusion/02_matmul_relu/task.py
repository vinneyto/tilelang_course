import tilelang
import tilelang.language as T


def build_matmul_relu(
    M: int,
    N: int,
    K: int,
    block_M: int = 32,
    block_N: int = 32,
    block_K: int = 16,
):
    """Скомпилировать Metal/CUDA-compatible fused tiled GEMM + ReLU."""
    raise NotImplementedError
