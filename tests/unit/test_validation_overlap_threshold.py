"""A mostly-disjoint sample_id overlap must fail, not warn.

Found 2026-09-10: `examples/ukbb/geosketch_phate/data/fit_labels.csv` belonged to
a superseded geosketch selection and overlapped the current one by 40.6%. The
pipeline ran to completion and published figures that coloured 40% of their
points, because the validator only raised at *zero* overlap and otherwise logged
a warning nobody read.

Zero overlap is a typo. Forty per cent is two datasets being confused for each
other, which is worse: it produces a plausible-looking result instead of an
error. The threshold below is what separates "some samples were filtered out",
which is normal, from that.
"""

import pytest

from manifold_genetics.utils.validation import ValidationError, validate_sample_id_overlap


def write_ids(path, ids):
    path.write_text("sample_id,value\n" + "".join(f"{i},1\n" for i in ids))
    return path


class TestOverlapThreshold:
    def test_identical_files_pass(self, tmp_path):
        a = write_ids(tmp_path / "a.csv", range(100))
        b = write_ids(tmp_path / "b.csv", range(100))

        validate_sample_id_overlap(a, b, "a", "b")

    def test_a_subset_passes(self, tmp_path):
        # The normal case: labels cover a superset of the embedded samples.
        a = write_ids(tmp_path / "a.csv", range(50))
        b = write_ids(tmp_path / "b.csv", range(100))

        validate_sample_id_overlap(a, b, "a", "b")

    def test_a_modest_shortfall_still_passes_with_a_warning(self, tmp_path, caplog):
        # 95 of 100: samples genuinely filtered out upstream. Normal.
        a = write_ids(tmp_path / "a.csv", range(100))
        b = write_ids(tmp_path / "b.csv", range(5, 100))

        validate_sample_id_overlap(a, b, "a", "b")

        assert any("mismatch" in r.message.lower() for r in caplog.records)

    def test_the_real_stale_label_case_is_rejected(self, tmp_path):
        # The geosketch incident: 24,355 of 60,000 = 40.6%.
        a = write_ids(tmp_path / "a.csv", range(60_000))
        b = write_ids(tmp_path / "b.csv", range(35_645, 95_645))

        with pytest.raises(ValidationError, match="40.6%|overlap"):
            validate_sample_id_overlap(a, b, "embedding", "labels")

    def test_zero_overlap_still_raises(self, tmp_path):
        a = write_ids(tmp_path / "a.csv", range(10))
        b = write_ids(tmp_path / "b.csv", range(100, 110))

        with pytest.raises(ValidationError):
            validate_sample_id_overlap(a, b, "a", "b")

    def test_the_error_reports_the_actual_percentage(self, tmp_path):
        a = write_ids(tmp_path / "a.csv", range(100))
        b = write_ids(tmp_path / "b.csv", range(90, 190))

        with pytest.raises(ValidationError) as exc:
            validate_sample_id_overlap(a, b, "a", "b")

        assert "10" in str(exc.value)

    def test_the_threshold_is_tunable_for_callers_that_expect_little_overlap(self, tmp_path):
        a = write_ids(tmp_path / "a.csv", range(100))
        b = write_ids(tmp_path / "b.csv", range(90, 190))

        validate_sample_id_overlap(a, b, "a", "b", min_overlap_fraction=0.05)
