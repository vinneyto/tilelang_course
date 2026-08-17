import tilelang


def candidate_configs() -> list[dict[str, int]]:
    """Вернуть небольшой осмысленный набор конфигураций GEMM."""
    raise NotImplementedError


def measure(kernel) -> float:
    """Измерить latency уже скомпилированного JITKernel в миллисекундах."""
    raise NotImplementedError

