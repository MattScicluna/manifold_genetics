"""Tests for the in-process metrics steps (pipeline/steps/metrics.py).

These steps replace `subprocess.run(["manifold-genetics", "metrics-*", ...])`.
The two behaviours that used to come free from going through the CLI — input
validation, and JSON round-tripping of the result dict — must survive the move,
so both are asserted here.
"""

import json
from pathlib import Path

import pytest

from manifold_genetics.pipeline.steps.metrics import (
    MetricsStepResult,
    run_admixture_metrics_step,
    run_geographic_metrics_step,
)

MODULE = "manifold_genetics.pipeline.steps.metrics"
_VALIDATORS = (
    "validate_embedding_csv",
    "validate_geographic_csv",
    "validate_admixture_csv",
    "validate_sample_id_overlap",
)


@pytest.fixture
def recorded_validators(monkeypatch):
    """Replace every validator with a recorder so the step runs on fake paths."""
    seen = {name: [] for name in _VALIDATORS}
    for name in _VALIDATORS:
        monkeypatch.setattr(
            f"{MODULE}.{name}",
            lambda *a, _n=name, **k: seen[_n].append((a, k)),
        )
    return seen


class TestGeographicMetricsStep:
    def test_writes_json_and_returns_parsed_values(
        self, tmp_path, monkeypatch, recorded_validators
    ):
        monkeypatch.setattr(
            f"{MODULE}.compute_geographic_preservation",
            lambda **k: {"correlation": 0.9, "p_value": 1e-3, "n_pairs": 12},
        )
        out = tmp_path / "metrics" / "geographic.json"

        result = run_geographic_metrics_step("emb.csv", "geo.csv", out)

        assert isinstance(result, MetricsStepResult)
        assert result.path == out
        assert result.skipped is False
        assert out.exists(), "parent directory must be created for the caller"
        assert json.loads(out.read_text())["correlation"] == 0.9
        assert result.values == json.loads(out.read_text())

    def test_forwards_options_to_the_compute_function(
        self, tmp_path, monkeypatch, recorded_validators
    ):
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_geographic_preservation",
            lambda **k: seen.update(k) or {"correlation": 0.1},
        )

        run_geographic_metrics_step(
            "emb.csv",
            "geo.csv",
            tmp_path / "g.json",
            longitude_col="lon",
            latitude_col="lat",
            num_samples=17,
            ignore_missing=False,
        )

        assert seen["longitude_col"] == "lon"
        assert seen["latitude_col"] == "lat"
        assert seen["num_samples"] == 17
        assert seen["ignore_missing"] is False

    def test_defaults_match_the_cli_defaults(self, tmp_path, monkeypatch, recorded_validators):
        """Drift here would silently change pipeline metrics vs. the CLI."""
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_geographic_preservation",
            lambda **k: seen.update(k) or {"correlation": 0.1},
        )

        run_geographic_metrics_step("emb.csv", "geo.csv", tmp_path / "g.json")

        assert seen["longitude_col"] == "longitude"
        assert seen["latitude_col"] == "latitude"
        assert seen["num_samples"] == 50000
        assert seen["ignore_missing"] is True

    def test_validates_inputs(self, tmp_path, monkeypatch, recorded_validators):
        monkeypatch.setattr(
            f"{MODULE}.compute_geographic_preservation", lambda **k: {"correlation": 0.1}
        )
        run_geographic_metrics_step("emb.csv", "geo.csv", tmp_path / "g.json")

        assert recorded_validators["validate_embedding_csv"], "embedding CSV must be validated"
        assert recorded_validators["validate_geographic_csv"], "geographic CSV must be validated"
        assert recorded_validators["validate_sample_id_overlap"], "sample overlap must be checked"


class TestAdmixtureMetricsStep:
    def test_builds_q_file_map_from_prefix_and_range(
        self, tmp_path, monkeypatch, recorded_validators
    ):
        """<prefix>.K.csv is the documented Q-file layout (spec constraint B)."""
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation",
            lambda **k: seen.update(k) or {},
        )
        prefix = tmp_path / "admixture" / "project"

        run_admixture_metrics_step("emb.csv", prefix, range(2, 5), tmp_path / "a.json")

        assert seen["q_files"] == {
            2: Path(f"{prefix}.2.csv"),
            3: Path(f"{prefix}.3.csv"),
            4: Path(f"{prefix}.4.csv"),
        }

    def test_json_round_trip_stringifies_k_keys(self, tmp_path, monkeypatch, recorded_validators):
        """compute_* returns int keys; the JSON artifact and run_pipeline()'s
        results dict have always had string keys. Preserve that."""
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation",
            lambda **k: {2: {"correlation": 0.5}, 3: {"correlation": 0.6}},
        )
        out = tmp_path / "a.json"

        result = run_admixture_metrics_step("emb.csv", tmp_path / "q", range(2, 4), out)

        assert set(result.values) == {"2", "3"}
        assert result.values["2"]["correlation"] == 0.5
        assert json.loads(out.read_text()) == result.values

    def test_forwards_options_to_the_compute_function(
        self, tmp_path, monkeypatch, recorded_validators
    ):
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation", lambda **k: seen.update(k) or {}
        )

        run_admixture_metrics_step(
            "emb.csv",
            tmp_path / "q",
            range(2, 4),
            tmp_path / "a.json",
            k_value=3,
            num_samples=11,
            subsample=100,
        )

        assert seen["k_value"] == 3
        assert seen["num_samples"] == 11
        assert seen["subsample"] == 100

    def test_defaults_match_the_cli_defaults(self, tmp_path, monkeypatch, recorded_validators):
        seen = {}
        monkeypatch.setattr(
            f"{MODULE}.compute_admixture_preservation", lambda **k: seen.update(k) or {}
        )

        run_admixture_metrics_step("emb.csv", tmp_path / "q", range(2, 4), tmp_path / "a.json")

        assert seen["k_value"] is None
        assert seen["num_samples"] == 50000
        assert seen["subsample"] is None

    def test_empty_k_range_raises(self, tmp_path, monkeypatch, recorded_validators):
        """An empty range would otherwise write an empty JSON and look successful."""
        monkeypatch.setattr(f"{MODULE}.compute_admixture_preservation", lambda **k: {})
        with pytest.raises(ValueError, match="[Nn]o admixture files"):
            run_admixture_metrics_step("emb.csv", tmp_path / "q", range(5, 5), tmp_path / "a.json")

    def test_validates_inputs_against_the_first_k(self, tmp_path, monkeypatch, recorded_validators):
        monkeypatch.setattr(f"{MODULE}.compute_admixture_preservation", lambda **k: {})
        prefix = tmp_path / "q"

        run_admixture_metrics_step("emb.csv", prefix, range(3, 6), tmp_path / "a.json")

        assert recorded_validators["validate_admixture_csv"]
        overlap_args = recorded_validators["validate_sample_id_overlap"][0][0]
        assert overlap_args[1] == f"{prefix}.3.csv", "overlap is checked against the lowest K"
