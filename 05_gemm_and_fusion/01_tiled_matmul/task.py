import tilelang
import tilelang.language as T


def build_matmul(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
):
    """Скомпилировать tiled float16 GEMM с float32 accumulator."""
    raise NotImplementedError

