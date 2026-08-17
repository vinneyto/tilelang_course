import tilelang
import tilelang.language as T


def build_matmul_relu(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
):
    """Скомпилировать fused tiled GEMM + ReLU."""
    raise NotImplementedError

