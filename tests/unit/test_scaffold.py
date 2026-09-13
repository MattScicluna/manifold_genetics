"""`manifold-genetics init` — the scaffolding a fresh install has nothing without.

`pip install manifold-genetics` ships no example config and no data: `examples/`
is in neither the wheel nor the sdist. Until this command existed the
documentation's opening move, `manifold-genetics run config.yaml`, was
unreachable for everyone who installed the package the documented way.
"""

import json

import pandas as pd
import pytest

from manifold_genetics.scaffold import hgdp_subsets, init_synthetic


class TestSyntheticScaffold:
    def test_writes_everything_a_run_needs(self, tmp_path):
        written = init_synthetic(tmp_path)

        for name in (
            "config.yaml",
            "data/fit_subset.bed",
            "data/fit_subset.bim",
            "data/fit_subset.fam",
            "data/project_subset.bed",
            "data/project_subset.bim",
            "data/project_subset.fam",
            "data/labels.csv",
            "colormap.json",
        ):
            assert (tmp_path / name).exists(), f"init did not write {name}"
        assert written == tmp_path / "config.yaml"

    def test_the_config_it_writes_is_accepted_by_the_loader(self, tmp_path):
        """The point of the command: what it writes must actually load."""
        from manifold_genetics.pipeline.configfile import load_config

        init_synthetic(tmp_path)
        kwargs = load_config(tmp_path / "config.yaml")

        assert kwargs["fit_plink"] == tmp_path / "data/fit_subset"
        assert kwargs["output_dir"] == tmp_path / "outputs"

    def test_labels_cover_every_genotyped_sample(self, tmp_path):
        """A label file that does not match its genotypes is this project's
        most expensive failure mode, so the one it ships cannot have it."""
        init_synthetic(tmp_path)

        fam = pd.read_csv(tmp_path / "data/project_subset.fam", sep=r"\s+", header=None, dtype=str)
        labels = pd.read_csv(tmp_path / "data/labels.csv", dtype=str)

        assert set(labels["sample_id"]) >= set(fam[1])

    def test_the_colormap_names_a_real_label_column(self, tmp_path):
        init_synthetic(tmp_path)

        colormap = json.loads((tmp_path / "colormap.json").read_text())
        labels = pd.read_csv(tmp_path / "data/labels.csv")

        for column, values in colormap.items():
            assert column in labels.columns, f"colormap names {column!r}, which is not a column"
            assert set(labels[column].astype(str)) <= set(values), "a label value has no colour"

    def test_refuses_to_overwrite_without_force(self, tmp_path):
        init_synthetic(tmp_path)

        with pytest.raises(FileExistsError, match="--force"):
            init_synthetic(tmp_path)

        init_synthetic(tmp_path, force=True)  # explicit is fine

    def test_the_bed_it_writes_is_readable(self, tmp_path):
        """It hand-encodes PLINK's 2-bit format, so round-tripping is the check."""
        from manifold_genetics.pca.plink import count_lines, read_bed_dosages

        init_synthetic(tmp_path)
        prefix = tmp_path / "data/fit_subset"
        n_samples = count_lines(f"{prefix}.fam")
        n_variants = count_lines(f"{prefix}.bim")

        dosages = read_bed_dosages(prefix, n_samples=n_samples, n_variants=n_variants)

        assert dosages.shape == (n_samples, n_variants)
        assert set(pd.unique(dosages.ravel())) <= {0.0, 1.0, 2.0}


class TestHgdpSubsets:
    """The two subsets are boolean filters on the cohort's metadata.

    Ported from examples/hgdp_1kgp/prepare_data.sh, which produced the subsets
    every published figure was made from, so the rule has to match it exactly:
    the fit set additionally excludes related samples, the project set does not.
    """

    @pytest.fixture
    def metadata(self):
        return pd.DataFrame(
            {
                "project_meta.sample_id": ["clean", "related", "outlier", "hard", "dirty"],
                "filter_pca_outlier": [False, False, True, False, False],
                "hard_filtered": [False, False, False, True, False],
                "filter_king_related": [False, True, False, False, False],
                "filter_contaminated": [False, False, False, False, True],
            }
        )

    def test_fit_excludes_related_samples(self, metadata):
        fit, _ = hgdp_subsets(metadata)

        assert list(fit) == ["clean"]

    def test_project_keeps_related_samples(self, metadata):
        _, project = hgdp_subsets(metadata)

        assert list(project) == ["clean", "related"]

    def test_both_drop_outliers_hard_filtered_and_contaminated(self, metadata):
        fit, project = hgdp_subsets(metadata)

        for excluded in ("outlier", "hard", "dirty"):
            assert excluded not in list(fit)
            assert excluded not in list(project)


class TestHgdpWithoutWorkingNetwork:
    """Fetching must be separable from preparing.

    Reported 2026-09-13 from a network with a TLS-intercepting proxy:

        Error: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate
        verify failed: self-signed certificate in certificate chain>

    The same shape of problem is routine on HPC compute nodes, which have no
    internet at all. In both cases the user can obtain the archive by some other
    means, so `init` has to be able to pick up from there -- and the message it
    prints has to say so, rather than telling them to retry the download that
    just failed.
    """

    @pytest.fixture
    def archive(self, tmp_path):
        """A stand-in for the real 183 MB tarball."""
        import tarfile

        raw = tmp_path / "src"
        raw.mkdir()
        (raw / "full_dataset.bed").write_bytes(bytes([0x6C, 0x1B, 0x01]))
        (raw / "full_dataset.bim").write_text("1 rs0 0 1 A G\n")
        (raw / "full_dataset.fam").write_text("s1 s1 0 0 0 -9\n")
        (raw / "metadata.csv").write_text(
            "project_meta.sample_id,filter_pca_outlier,hard_filtered,"
            "filter_king_related,filter_contaminated\ns1,False,False,False,False\n"
        )
        path = tmp_path / "hgdp_1kgp_full.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for f in sorted(raw.iterdir()):
                tar.add(f, arcname=f.name)
        return path

    def test_an_archive_already_downloaded_is_extracted(self, tmp_path, archive, monkeypatch):
        """--no-download must mean "do not fetch", not "do not unpack"."""
        import shutil

        out = tmp_path / "out"
        (out / "data").mkdir(parents=True)
        shutil.copy(archive, out / "data" / archive.name)

        monkeypatch.setattr(
            "manifold_genetics.scaffold._run_plink2_keep",
            lambda *a, **k: None,
        )
        from manifold_genetics.scaffold import init_hgdp

        init_hgdp(out, download=False)

        assert (out / "data" / "raw" / "full_dataset.bed").exists()

    def test_an_archive_elsewhere_can_be_named(self, tmp_path, archive, monkeypatch):
        monkeypatch.setattr(
            "manifold_genetics.scaffold._run_plink2_keep",
            lambda *a, **k: None,
        )
        from manifold_genetics.scaffold import init_hgdp

        init_hgdp(tmp_path / "out", archive=archive)

        assert (tmp_path / "out" / "data" / "raw" / "metadata.csv").exists()

    def test_a_failed_download_explains_the_way_round_it(self, tmp_path, monkeypatch):
        import urllib.request

        def _tls_failure(url, dest):
            raise OSError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate")

        monkeypatch.setattr(urllib.request, "urlretrieve", _tls_failure)
        from manifold_genetics.scaffold import init_hgdp

        with pytest.raises(RuntimeError) as excinfo:
            init_hgdp(tmp_path / "out")

        message = str(excinfo.value)
        assert "--archive" in message, "the error must name the way to supply the file"
        assert "hgdp_1kgp_full.tar.gz" in message, "the error must name what to download"
