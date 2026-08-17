import pytest

pytest.importorskip("tilelang")
from task import candidate_configs, measure


def test_candidate_configs_are_meaningful():
    configs = candidate_configs()
    required = {"block_M", "block_N", "block_K", "num_stages", "threads"}
    assert len(configs) >= 3
    assert all(set(config) == required for config in configs)
    assert all(all(isinstance(v, int) and v > 0 for v in config.values()) for config in configs)
    assert len({tuple(sorted(config.items())) for config in configs}) == len(configs)


def test_measure_uses_profiler_contract():
    class FakeProfiler:
        def do_bench(self):
            return 0.125

    class FakeKernel:
        def get_profiler(self, **kwargs):
            return FakeProfiler()

    assert measure(FakeKernel()) == pytest.approx(0.125)

