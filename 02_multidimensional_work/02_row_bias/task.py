import tilelang
import tilelang.language as T


def build_row_bias_relu(
    M: int, N: int, block_M: int = 16, block_N: int = 16
):
    """Скомпилировать Y[row, col] = relu(X[row, col] + bias[col])."""
    raise NotImplementedError

