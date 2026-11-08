"""Exact PHATE refuses, up front, a run that cannot fit in memory (#75).

Without landmarks PHATE holds dense n x n operators. On All of Us v9 the CLI's
`embed` (which defaults to no landmarks) was OOM-killed at "Transforming" on
101,142 samples after the fit had already succeeded. Exact stays the default;
only a run that would be killed anyway is stopped, at the start, with the remedy.
"""

import numpy as np
import pytest

from manifold_genetics.embeddings import phate as phate_module
from manifold_genetics.embeddings.phate import PHATE
from manifold_genetics.utils import memory


@pytest.fixture
def available(monkeypatch):
    def set_gb(gb):
        monkeypatch.setattr(memory, "available_memory_bytes", lambda: int(gb * 1024**3))

    return set_gb


class TestExactGuard:
    def test_an_exact_run_that_cannot_fit_is_refused_with_the_remedy(self, available):
        available(1)  # 3 x 8 x 20,000^2 bytes is ~9 GB
        model = PHATE(n_landmark=None)

        with pytest.raises(MemoryError, match="--n-landmark 10000 --random-landmarking"):
            model._check_exact_fits_in_memory(20_000)

    def test_an_exact_run_that_fits_goes_ahead(self, available):
        available(64)
        PHATE(n_landmark=None)._check_exact_fits_in_memory(20_000)

    def test_landmarked_runs_are_never_checked(self, available):
        available(0.001)
        PHATE(n_landmark=10_000)._check_exact_fits_in_memory(1_000_000)

    def test_the_override_runs_exact_anyway(self, available, monkeypatch):
        available(0.001)
        monkeypatch.setenv("MANIFOLD_GENETICS_FORCE_EXACT", "1")
        PHATE(n_landmark=None)._check_exact_fits_in_memory(1_000_000)

    def test_unknown_memory_is_not_a_reason_to_refuse(self, monkeypatch):
        monkeypatch.setattr(memory, "available_memory_bytes", lambda: None)
        PHATE(n_landmark=None)._check_exact_fits_in_memory(1_000_000)

    def test_fit_checks_before_phate_does_any_work(self, available, monkeypatch):
        available(0.0001)
        model = PHATE(n_landmark=None)
        called = []
        monkeypatch.setattr(model.model, "fit", lambda X: called.append(X))

        with pytest.raises(MemoryError):
            model.fit(np.zeros((5_000, 3)))
        assert not called

    def test_fit_transform_checks_too(self, available, monkeypatch):
        available(0.0001)
        model = PHATE(n_landmark=None)
        monkeypatch.setattr(model.model, "fit_transform", lambda X: pytest.fail("ran"))

        with pytest.raises(MemoryError):
            model.fit_transform(np.zeros((5_000, 3)))

    def test_the_estimate_is_three_dense_copies(self):
        assert phate_module.PHATE.EXACT_DENSE_COPIES == 3


class TestAvailableMemory:
    """cgroup v2 limits, read from the process's cgroup up to the root."""

    def _tree(self, tmp_path, limits):
        """limits: {relative cgroup path: (memory.max, memory.current)}"""
        root = tmp_path / "cgroup"
        for rel, (mx, cur) in limits.items():
            d = root / rel if rel else root
            d.mkdir(parents=True, exist_ok=True)
            (d / "memory.max").write_text(f"{mx}\n")
            (d / "memory.current").write_text(f"{cur}\n")
        return root

    def test_the_job_limit_is_used_not_the_node(self, tmp_path):
        gib = 1024**3
        root = self._tree(
            tmp_path,
            {
                "": ("max", 0),
                "slurm/job_1": (int(62.5 * gib), 2 * gib),
                "slurm/job_1/step/user/task_0": ("max", 0),
            },
        )
        proc = tmp_path / "cgroup.proc"
        proc.write_text("0::/slurm/job_1/step/user/task_0\n")
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(f"MemAvailable:   {229 * 1024**2} kB\n")  # a 251 GB node

        got = memory.available_memory_bytes(proc, root, meminfo)

        assert got == int(62.5 * gib) - 2 * gib

    def test_without_a_cgroup_limit_memavailable_is_used(self, tmp_path):
        root = self._tree(tmp_path, {"": ("max", 0)})
        proc = tmp_path / "cgroup.proc"
        proc.write_text("0::/\n")
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal: 1 kB\nMemAvailable:   1048576 kB\n")

        assert memory.available_memory_bytes(proc, root, meminfo) == 1024**3

    def test_nothing_readable_falls_back_to_physical_memory(self, tmp_path):
        got = memory.available_memory_bytes(
            tmp_path / "missing", tmp_path / "missing", tmp_path / "missing"
        )

        assert got is None or got > 0
