import pytest
import torch

pytest.importorskip("tilelang")
from .task import candidate_configs, measure


def test_candidate_configs_are_meaningful():
    configs = candidate_configs()
    required = {"block_M", "block_N", "block_K", "num_stages", "threads"}
    assert len(configs) >= 3
    assert all(set(config) == required for config in configs)
    assert all(all(isinstance(v, int) for v in config.values()) for config in configs)
    assert all(config["num_stages"] >= 0 for config in configs)
    positive_keys = required - {"num_stages"}
    assert all(all(config[key] > 0 for key in positive_keys) for config in configs)
    assert len({tuple(sorted(config.items())) for config in configs}) == len(configs)


def test_measure_runs_warmup_and_repetitions():
    class FakeKernel:
        calls = 0

        def __call__(self, *args):
            self.calls += 1

    kernel = FakeKernel()
    latency = measure(kernel, (), torch.device("cpu"), warmup=2, repeat=3)
    assert latency > 0
    assert kernel.calls == 5
