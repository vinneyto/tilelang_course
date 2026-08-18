import tilelang
import torch


def candidate_configs() -> list[dict[str, int]]:
    """Вернуть небольшой осмысленный набор конфигураций GEMM."""
    raise NotImplementedError


def measure(
    kernel,
    args: tuple[torch.Tensor, ...],
    device: torch.device,
    warmup: int = 5,
    repeat: int = 20,
) -> float:
    """Переносимо измерить среднюю latency kernel в миллисекундах."""
    raise NotImplementedError
