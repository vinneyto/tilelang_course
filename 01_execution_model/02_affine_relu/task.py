import tilelang
import tilelang.language as T


def build_affine_relu(
    N: int, scale: float, bias: float, block: int = 256
):
    """Скомпилировать fused-ядро Y = max(X * scale + bias, 0)."""
    raise NotImplementedError

