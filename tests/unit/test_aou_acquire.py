"""acquire_aou with GCS and BigQuery replaced by stubs that leave behind what the
real ones do. Untestable for real until workbench access returns (#124); these
pin the sequence and the files.

The port is of examples/aou/shared/download_aou_data.sh plus the project-labels
block of examples/aou/hgdp_1kgp_proj/prepare_data.sh (step 14). What each test
pins is named against the line of shell it came from.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from manifold_genetics import aou

# What the bucket's arrays.fam looks like: the script's awk keeps $2..$5 and
# replaces $1 (FID) with AOU and $6 (phenotype) with -9.
_BUCKET_FAM = "0 1000001 0 0 1 0\n0 1000002 0 0 2 0\n0 1000003 0 0 0 0\n"


@pytest.fixture
def workbench(monkeypatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "proj")
    monkeypatch.setenv("WORKSPACE_CDR", "cdr.v8")
    monkeypatch.setattr(aou, "aou_environment_problems", lambda: [])


def _runner_factory(calls):
    """A subprocess.run stand-in: records argv and, for `gsutil cp`, writes the
    destination the way the real copy would."""

    def runner(argv, **kw):
        calls.append(list(argv))
        if argv[0] == "gsutil" and "cp" in argv:
            src, dest = argv[-2], Path(argv[-1])
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.endswith(".fam"):
                dest.write_text(_BUCKET_FAM)
            elif src.endswith(".bim"):
                dest.write_text("1\trs1\t0\t1\tA\tG\n")
            else:
                dest.write_bytes(b"\x6c\x1b\x01")
        return subprocess.CompletedProcess(argv, 0)

    return runner


def _query_factory(queries, person=None):
    """A pandas_gbq.read_gbq stand-in: records SQL and answers each query."""
    if person is None:
        person = pd.DataFrame(
            {
                "person_id": [1000001, 1000002, 1000003, 1000004],
                "gender": ["Female", "Male", "Female", "Male"],
                "race": ["White", "Asian", "PMI: Skip", "Black"],
                "ethnicity": [
                    "Not Hispanic or Latino",
                    "PMI: Skip",
                    "Hispanic or Latino",
                    "Not Hispanic or Latino",
                ],
            }
        )

    def query(sql, **kw):
        queries.append(sql)
        if "zip3_ses_map" in sql:
            return pd.DataFrame({"person_id": [1000001, 1000002], "zip_code": ["100", "200"]})
        return person.copy()

    return query


def _acquire(tmp_path, calls=None, queries=None, **kw):
    calls = [] if calls is None else calls
    queries = [] if queries is None else queries
    return aou.acquire_aou(
        tmp_path, runner=_runner_factory(calls), query=_query_factory(queries), **kw
    )


class TestWholeCohortDirectory:
    def test_the_config_is_returned_and_loads_as_a_whole_cohort(self, tmp_path, workbench):
        from manifold_genetics.pipeline.configfile import load_config

        config = _acquire(tmp_path)

        assert config == tmp_path / "config.yaml"
        kwargs = load_config(config)
        assert kwargs["fit_plink"] == kwargs["project_plink"]
        assert kwargs["fit_plink"] == (tmp_path / "data/project_subset").resolve()
        assert kwargs["n_pcs"] == 20
        assert (
            kwargs["embedding_params"]["embed_batch_size"] == 60000
        ), "the batch size bounds memory on 400k+ samples"
        assert kwargs["skip_admixture"] is True

    def test_the_config_says_how_to_project_it_onto_hgdp(self, tmp_path, workbench):
        """The old comment said the genotypes were not fetched; now they are,
        and the comment has to say what the cohort alone is for."""
        text = _acquire(tmp_path).read_text()

        assert "preset: whole_cohort" in text
        assert "not fetched" not in text.lower()
        assert "acquire hgdp --archive gs://" in text
        assert "--preset harmonise --fit-has-chr-prefix" in text
        assert "projection_plot" not in text, "those keys are for projections; preprocess sets them"

    def test_project_subset_links_to_the_fam_fixed_download(self, tmp_path, workbench):
        """The .bed is large: linked, not copied. Absolute targets so the link
        survives being read from anywhere."""
        _acquire(tmp_path)

        for ext in ("bed", "bim", "fam"):
            link = tmp_path / f"data/project_subset.{ext}"
            assert link.is_symlink(), ext
            target = link.readlink()
            assert target.is_absolute()
            assert target == (tmp_path / f"data/raw/extractedChrAll.{ext}").resolve()
            assert link.exists()


class TestDownload:
    """download_aou_data.sh L109-127: one `gsutil -u $GOOGLE_PROJECT cp` per
    extension, arrays.* in the bucket renamed to extractedChrAll.* on disk."""

    def test_each_extension_is_copied_under_the_billing_project(self, tmp_path, workbench):
        calls = []
        _acquire(tmp_path, calls=calls)

        gsutil = [c for c in calls if c[0] == "gsutil"]
        assert [c[1:3] for c in gsutil] == [["-u", "proj"]] * 3
        assert [c[-2] for c in gsutil] == [
            f"gs://fc-aou-datasets-controlled/v8/microarray/plink/arrays.{ext}"
            for ext in ("bed", "bim", "fam")
        ]
        assert [c[-1] for c in gsutil] == [
            str(tmp_path / f"data/raw/extractedChrAll.{ext}") for ext in ("bed", "bim", "fam")
        ]

    def test_only_what_is_missing_is_fetched(self, tmp_path, workbench):
        raw = tmp_path / "data/raw"
        raw.mkdir(parents=True)
        (raw / "extractedChrAll.bed").write_bytes(b"\x6c\x1b\x01")

        calls = []
        _acquire(tmp_path, calls=calls)

        fetched = [c[-2] for c in calls if c[0] == "gsutil"]
        assert fetched == [
            "gs://fc-aou-datasets-controlled/v8/microarray/plink/arrays.bim",
            "gs://fc-aou-datasets-controlled/v8/microarray/plink/arrays.fam",
        ]

    def test_nothing_runs_plink(self, tmp_path, workbench):
        """The script's per-chromosome split (L146-156) is not ported: nothing
        downstream reads extractedChr{1..22}."""
        calls = []
        _acquire(tmp_path, calls=calls)

        assert {c[0] for c in calls} == {"gsutil"}
        assert not list((tmp_path / "data/raw").glob("extractedChr1.*"))

    def test_a_failed_copy_is_not_swallowed(self, tmp_path, workbench):
        def runner(argv, **kw):
            raise subprocess.CalledProcessError(1, argv)

        with pytest.raises(subprocess.CalledProcessError):
            aou.acquire_aou(tmp_path, runner=runner, query=_query_factory([]))
        assert not (tmp_path / "config.yaml").exists()

    def test_a_missing_gsutil_binary_is_a_friendly_error(self, tmp_path, workbench):
        def runner(argv, **kw):
            raise FileNotFoundError(2, "No such file or directory", "gsutil")

        with pytest.raises(RuntimeError, match="gsutil"):
            aou.acquire_aou(tmp_path, runner=runner, query=_query_factory([]))
        assert not (tmp_path / "config.yaml").exists()


class TestFamFix:
    """download_aou_data.sh L134-144: `awk '{print "AOU\\t"$2"\\t"$3"\\t"$4"\\t"$5"\\t-9"}'`
    over the original, which is kept as extractedChrAll.Original.fam."""

    def test_fid_becomes_aou_and_phenotype_minus_nine(self, tmp_path, workbench):
        _acquire(tmp_path)

        raw = tmp_path / "data/raw"
        assert (raw / "extractedChrAll.Original.fam").read_text() == _BUCKET_FAM
        assert (raw / "extractedChrAll.fam").read_text() == (
            "AOU\t1000001\t0\t0\t1\t-9\nAOU\t1000002\t0\t0\t2\t-9\nAOU\t1000003\t0\t0\t0\t-9\n"
        )

    def test_the_fix_is_not_applied_twice(self, tmp_path, workbench):
        """Once .Original.fam exists the download is never renamed again, so
        the original survives every run and the .fam is the same fixed form."""
        _acquire(tmp_path)
        raw = tmp_path / "data/raw"
        fixed = (raw / "extractedChrAll.fam").read_text()

        _acquire(tmp_path, force=True)

        assert (raw / "extractedChrAll.Original.fam").read_text() == _BUCKET_FAM
        assert (raw / "extractedChrAll.fam").read_text() == fixed

    def test_an_orphaned_marker_does_not_let_a_raw_fam_through(self, tmp_path, workbench):
        """Interrupted between the rename and the rewrite (or the .fam deleted):
        the marker is present, the .fam is missing, so the next run downloads
        the raw .fam again. The script would then have skipped the fix and
        linked an unfixed .fam. The original is the source of truth; the .fam
        is regenerated from it every time."""
        _acquire(tmp_path)
        raw = tmp_path / "data/raw"
        (raw / "extractedChrAll.fam").unlink()
        (tmp_path / "config.yaml").unlink()

        calls = []
        _acquire(tmp_path, calls=calls)

        assert [c[-2] for c in calls if c[0] == "gsutil"] == [
            "gs://fc-aou-datasets-controlled/v8/microarray/plink/arrays.fam"
        ], "the raw .fam was fetched again"
        assert (raw / "extractedChrAll.fam").read_text().startswith("AOU\t1000001\t0\t0\t1\t-9")
        assert (raw / "extractedChrAll.Original.fam").read_text() == _BUCKET_FAM
        assert (tmp_path / "data/project_subset.fam").read_text().startswith("AOU\t")


class TestMetadata:
    """download_aou_data.sh L190-273, the embedded extract_metadata.py."""

    def test_both_queries_run_against_the_workspace_cdr(self, tmp_path, workbench):
        queries = []
        _acquire(tmp_path, queries=queries)

        assert len(queries) == 2
        assert "`cdr.v8.person` person" in queries[0]
        assert "`cdr.v8.zip3_ses_map` zip_code" in queries[1]
        assert "observation_source_concept_id = 1585250" in queries[1]

    def test_the_two_tsvs_are_written(self, tmp_path, workbench):
        _acquire(tmp_path)

        meta = tmp_path / "data/raw/Metadata"
        demographics = pd.read_csv(meta / "DemographicData.tsv", sep="\t")
        ses = pd.read_csv(meta / "SocioeconomicZipCodes.tsv", sep="\t")
        assert len(demographics) == 4
        assert list(ses.columns) == ["person_id", "zip_code"]

    def test_race_ethnicity_is_derived_as_the_script_does(self, tmp_path, workbench):
        """L229-242: uninformative race and ethnicity answers become "No
        information", and race_ethnicity is race except that a "No information"
        race with Hispanic or Latino ethnicity is "Hispanic or Latino"."""
        _acquire(tmp_path)

        demographics = pd.read_csv(tmp_path / "data/raw/Metadata/DemographicData.tsv", sep="\t")
        by_id = demographics.set_index("person_id")
        assert by_id.loc[1000003, "race"] == "No information"
        assert by_id.loc[1000003, "race_ethnicity"] == "Hispanic or Latino"
        assert by_id.loc[1000002, "ethnicity"] == "No information"
        assert by_id.loc[1000002, "race_ethnicity"] == "Asian"
        assert by_id.loc[1000001, "race_ethnicity"] == "White"

    def test_existing_metadata_is_not_queried_again(self, tmp_path, workbench):
        """L167-170: both files present means no BigQuery."""
        _acquire(tmp_path)

        queries = []
        _acquire(tmp_path, queries=queries, force=True)

        assert queries == []

    def test_the_default_query_needs_the_aou_extra(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "pandas_gbq", None)

        with pytest.raises(ImportError, match=r"\[aou\]"):
            aou._read_gbq("SELECT 1")


class TestLabels:
    """prepare_data.sh step 14, the project-labels block: person_id becomes
    sample_id, rows are those in the .fam, and the race columns are kept."""

    def test_labels_are_the_fam_samples_with_their_race_columns(self, tmp_path, workbench):
        _acquire(tmp_path)

        labels = pd.read_csv(tmp_path / "data/labels.csv", dtype=str)
        assert list(labels.columns) == ["sample_id", "race_ethnicity", "race", "ethnicity"], (
            "race_ethnicity first: it becomes the first colormap key, which "
            "preprocess picks as the projection's plot column"
        )
        assert sorted(labels["sample_id"]) == ["1000001", "1000002", "1000003"]
        assert "1000004" not in set(labels["sample_id"]), "not in the .fam"

    def test_the_colormap_covers_every_label_value(self, tmp_path, workbench):
        _acquire(tmp_path)

        colormap = json.loads((tmp_path / "colormap.json").read_text())
        assert list(colormap) == ["race_ethnicity", "race", "ethnicity"]
        assert set(colormap["race_ethnicity"]) == {"White", "Asian", "Hispanic or Latino"}

    def test_labels_are_rewritten_every_run(self, tmp_path, workbench):
        """Reusing labels because the file exists is how ukbb/geosketch_phate's
        went stale; the .fam and metadata are on disk, so rewriting is cheap."""
        _acquire(tmp_path)
        (tmp_path / "data/labels.csv").write_text("sample_id,race_ethnicity\nstale,x\n")

        _acquire(tmp_path, force=True)

        labels = pd.read_csv(tmp_path / "data/labels.csv", dtype=str)
        assert "stale" not in set(labels["sample_id"])

    def test_samples_without_metadata_are_counted(self, tmp_path, workbench, caplog):
        person = pd.DataFrame(
            {"person_id": [1000001, 1000002], "race": ["White", "Asian"], "ethnicity": ["x", "y"]}
        )
        calls, queries = [], []
        with caplog.at_level("WARNING"):
            aou.acquire_aou(
                tmp_path,
                runner=_runner_factory(calls),
                query=_query_factory(queries, person=person),
            )

        assert "1 samples" in caplog.text
        labels = pd.read_csv(tmp_path / "data/labels.csv", dtype=str)
        assert list(labels["sample_id"]) == ["1000001", "1000002"]

    def test_metadata_covering_under_half_the_cohort_is_refused(self, tmp_path, workbench):
        """The rule `acquire custom` applies to a label file it is handed, so a
        wrong-CDR query result cannot become a label file that colours nothing."""
        person = pd.DataFrame({"person_id": [1000001], "race": ["White"], "ethnicity": ["x"]})

        with pytest.raises(ValueError, match=r"DemographicData\.tsv.*33\.3%.*Delete it"):
            aou.acquire_aou(
                tmp_path, runner=_runner_factory([]), query=_query_factory([], person=person)
            )
        assert not (tmp_path / "config.yaml").exists()
        assert not (tmp_path / "data/labels.csv").exists()

    @pytest.mark.parametrize(
        "content", ["", "person_id\trace\tethnicity\trace_ethnicity\n"], ids=["empty", "header"]
    )
    def test_an_empty_demographics_file_is_refused_not_reused(self, tmp_path, workbench, content):
        """A zero-byte or header-only DemographicData.tsv (crash mid-write, the
        wrong CDR) is "present" to the metadata step; it must fail here with
        the file named, not with pandas' EmptyDataError, and not succeed."""
        meta = tmp_path / "data/raw/Metadata"
        meta.mkdir(parents=True)
        (meta / "DemographicData.tsv").write_text(content)
        (meta / "SocioeconomicZipCodes.tsv").write_text("person_id\tzip_code\n")

        queries = []
        with pytest.raises(ValueError, match=r"DemographicData\.tsv.*0\.0%.*Delete it"):
            _acquire(tmp_path, queries=queries)
        assert queries == [], "both files were present, so nothing was queried"
        assert not (tmp_path / "config.yaml").exists()


class TestRefusals:
    def test_outside_the_workbench_it_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(aou, "aou_environment_problems", lambda: ["GOOGLE_PROJECT is not set"])

        with pytest.raises(EnvironmentError, match="GOOGLE_PROJECT"):
            aou.acquire_aou(tmp_path)

        assert not list(tmp_path.iterdir())

    def test_an_existing_config_is_not_overwritten_without_force(self, tmp_path, workbench):
        _acquire(tmp_path)

        with pytest.raises(FileExistsError):
            _acquire(tmp_path)

    def test_the_scaffold_name_is_the_same_function(self):
        from manifold_genetics import scaffold

        assert scaffold.acquire_aou is aou.acquire_aou
        assert scaffold.aou_environment_problems is aou.aou_environment_problems


class TestCli:
    def test_a_failed_gsutil_names_the_command(self, tmp_path, monkeypatch, capsys):
        from manifold_genetics import cli, scaffold

        def boom(out, force=False):
            raise subprocess.CalledProcessError(1, ["gsutil", "-u", "proj", "cp", "a", "b"])

        monkeypatch.setattr(scaffold, "acquire_aou", boom)

        assert cli.main(["acquire", "aou", "--out", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert "gsutil -u proj cp a b" in err
