"""`manifold-genetics acquire` — the scaffolding a fresh install has nothing without.

`pip install manifold-genetics` ships no example config and no data: `examples/`
is in neither the wheel nor the sdist. Until this command existed the
documentation's opening move, `manifold-genetics run config.yaml`, was
unreachable for everyone who installed the package the documented way.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.scaffold import (
    DLA_TREE_EDGES,
    DLA_TREE_GAPS,
    _tree_layout,
    acquire_synthetic,
    dla_tree,
    genotypes_from_coordinates,
    hgdp_subsets,
    plot_dla_tree,
)


class TestDlaTree:
    """The cohort lies along the tree in manylatents' `dla_tree_from_graph.yaml`.

    The generator is a port, not an import, and was checked against the
    original array for array (seed 42, 2026-09-13). These guard the properties
    that check relied on, so a later edit cannot quietly break the match.
    """

    def test_one_sample_per_position_on_each_data_edge(self):
        coordinates, branch = dla_tree()

        data_edges = [edge for edge in DLA_TREE_EDGES if edge[2] not in DLA_TREE_GAPS]
        assert coordinates.shape == (sum(edge[3] for edge in data_edges), 100)
        counts = dict(zip(*np.unique(branch, return_counts=True)))
        assert counts == {edge[2]: edge[3] for edge in data_edges}

    def test_is_reproducible(self):
        first, _ = dla_tree()
        second, _ = dla_tree()

        assert np.array_equal(first, second)

    def test_a_gap_edge_leaves_a_gap(self):
        """Edge 2 continues edge 1 at node 2, so it starts where edge 1 ends.
        Edge 3 starts at node 8, on the far side of a gap edge from node 2, so
        it does not -- that gap is the point of having gap edges."""
        coordinates, branch = dla_tree(sigma=0.0)

        end_of_1 = coordinates[branch == 1][-1]
        step = np.linalg.norm(np.diff(coordinates[branch == 1], axis=0), axis=1).max()

        assert np.allclose(coordinates[branch == 2][0], end_of_1)
        assert np.linalg.norm(coordinates[branch == 3][0] - end_of_1) > 10 * step

    def test_layout_puts_siblings_side_by_side_under_their_parent(self):
        """Node 2 has three children (edges 2, 6 and 12) and node 6 has two
        (edges 5 and 8): they must sit one row below, spread around the
        parent, so the drawing reads as the tree rather than as a line."""
        position = _tree_layout(DLA_TREE_EDGES)

        assert position[1] == (0.0, 0.0)
        assert [position[n] for n in (3, 10, 8)] == [(-3.0, -4.0), (0.0, -4.0), (3.0, -4.0)]
        assert position[7][1] == position[9][1] < position[6][1]
        assert position[7][0] < position[6][0] < position[9][0]

    def test_draws_a_png(self, tmp_path):
        written = plot_dla_tree(tmp_path / "tree.png")

        assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_genotypes_are_plink_dosages(self):
        coordinates, _ = dla_tree()

        dosages = genotypes_from_coordinates(coordinates, n_variants=50)

        assert dosages.shape == (len(coordinates), 50)
        assert set(np.unique(dosages)) <= {0, 1, 2}


class TestSyntheticScaffold:
    def test_writes_everything_a_run_needs(self, tmp_path):
        written = acquire_synthetic(tmp_path)

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
            "dla_tree_ground_truth.png",
        ):
            assert (tmp_path / name).exists(), f"acquire did not write {name}"
        assert written == tmp_path / "config.yaml"

    def test_the_config_it_writes_is_accepted_by_the_loader(self, tmp_path):
        """The point of the command: what it writes must actually load."""
        from manifold_genetics.pipeline.configfile import load_config

        acquire_synthetic(tmp_path)
        kwargs = load_config(tmp_path / "config.yaml")

        assert kwargs["fit_plink"] == tmp_path / "data/fit_subset"
        assert kwargs["output_dir"] == tmp_path / "outputs"

    def test_labels_cover_every_genotyped_sample(self, tmp_path):
        """A label file that does not match its genotypes is this project's
        most expensive failure mode, so the one it ships cannot have it."""
        acquire_synthetic(tmp_path)

        fam = pd.read_csv(tmp_path / "data/project_subset.fam", sep=r"\s+", header=None, dtype=str)
        labels = pd.read_csv(tmp_path / "data/labels.csv", dtype=str)

        assert set(labels["sample_id"]) >= set(fam[1])

    def test_the_colormap_names_a_real_label_column(self, tmp_path):
        acquire_synthetic(tmp_path)

        colormap = json.loads((tmp_path / "colormap.json").read_text())
        labels = pd.read_csv(tmp_path / "data/labels.csv")

        for column, values in colormap.items():
            assert column in labels.columns, f"colormap names {column!r}, which is not a column"
            assert set(labels[column].astype(str)) <= set(values), "a label value has no colour"

    def test_refuses_to_overwrite_without_force(self, tmp_path):
        acquire_synthetic(tmp_path)

        with pytest.raises(FileExistsError, match="--force"):
            acquire_synthetic(tmp_path)

        acquire_synthetic(tmp_path, force=True)  # explicit is fine

    def test_the_bed_it_writes_is_readable(self, tmp_path):
        """It hand-encodes PLINK's 2-bit format, so round-tripping is the check."""
        from manifold_genetics.pca.plink import count_lines, read_bed_dosages

        acquire_synthetic(tmp_path)
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
    means, so `acquire` has to be able to pick up from there -- and the message it
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
            "project_meta.sample_id,Genetic_region_merged,filter_pca_outlier,"
            "hard_filtered,filter_king_related,filter_contaminated\n"
            "s1,Africa,False,False,False,False\n"
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
        from manifold_genetics.scaffold import acquire_hgdp

        acquire_hgdp(out, download=False)

        assert (out / "data" / "raw" / "full_dataset.bed").exists()

    def test_an_archive_elsewhere_can_be_named(self, tmp_path, archive, monkeypatch):
        monkeypatch.setattr(
            "manifold_genetics.scaffold._run_plink2_keep",
            lambda *a, **k: None,
        )
        from manifold_genetics.scaffold import acquire_hgdp

        acquire_hgdp(tmp_path / "out", archive=archive)

        assert (tmp_path / "out" / "data" / "raw" / "metadata.csv").exists()

    def test_a_failed_download_explains_the_way_round_it(self, tmp_path, monkeypatch):
        import urllib.request

        def _tls_failure(url, dest):
            raise OSError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate")

        monkeypatch.setattr(urllib.request, "urlretrieve", _tls_failure)
        # curl and wget must be blocked too, or this reaches the real network:
        # an earlier version of this test downloaded 74 MB before it was killed.
        from manifold_genetics.utils import tools

        monkeypatch.setattr(tools.shutil, "which", lambda _: None)
        from manifold_genetics.scaffold import acquire_hgdp

        with pytest.raises(RuntimeError) as excinfo:
            acquire_hgdp(tmp_path / "out")

        message = str(excinfo.value)
        assert "--archive" in message, "the error must name the way to supply the file"
        assert "hgdp_1kgp_full.tar.gz" in message, "the error must name what to download"


class TestHgdpLayouts:
    """The workbench ships a different HGDP+1KGP archive from the public one.

    The public Dropbox archive unpacks to `full_dataset.*` plus a `metadata.csv`
    with QC and relatedness flags. The archive kept beside All of Us unpacks to
    `extractedChrAllUnpruned.*`: no metadata, chromosomes without a `chr` prefix,
    and the population carried in the FID as `forReference<Population>`.
    `examples/aou/hgdp_1kgp_proj/prepare_data.sh` fixed both by hand; `acquire
    hgdp` has to recognise which one it unpacked and do the same.
    """

    def _workbench_archive(self, tmp_path, subdir=None):
        import tarfile

        raw = tmp_path / "src"
        raw.mkdir()
        (raw / "extractedChrAllUnpruned.bim").write_text(
            "1\trs1\t0\t100\tA\tG\n22\trs2\t0\t200\tC\tT\n"
        )
        (raw / "extractedChrAllUnpruned.fam").write_text(
            "forReferenceYoruba\tS1\t0\t0\t0\t-9\nforReferenceFrench\tS2\t0\t0\t0\t-9\n"
        )
        (raw / "extractedChrAllUnpruned.bed").write_bytes(b"\x6c\x1b\x01\x00\x00")
        archive = tmp_path / "1KGPHGDP.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for f in raw.iterdir():
                tar.add(f, arcname=f"{subdir}/{f.name}" if subdir else f.name)
        return archive

    def test_detects_the_public_layout(self, tmp_path):
        from manifold_genetics.scaffold import _detect_hgdp_layout

        (tmp_path / "full_dataset.bed").write_bytes(b"")
        assert _detect_hgdp_layout(tmp_path) == "public"

    def test_detects_the_workbench_layout(self, tmp_path):
        from manifold_genetics.scaffold import _detect_hgdp_layout

        (tmp_path / "extractedChrAllUnpruned.bed").write_bytes(b"")
        assert _detect_hgdp_layout(tmp_path) == "workbench"

    def test_an_unknown_layout_names_both_it_looked_for(self, tmp_path):
        from manifold_genetics.scaffold import _detect_hgdp_layout

        with pytest.raises(FileNotFoundError, match="full_dataset.*extractedChrAllUnpruned"):
            _detect_hgdp_layout(tmp_path)

    def test_the_workbench_archive_is_normalised_to_the_public_layout(self, tmp_path):
        from manifold_genetics.scaffold import _extract_hgdp_archive

        raw = tmp_path / "raw"
        _extract_hgdp_archive(self._workbench_archive(tmp_path), raw)
        bim = (raw / "full_dataset.bim").read_text().splitlines()
        assert bim[0].startswith("chr1\t") and bim[1].startswith("chr22\t")
        fam = (raw / "full_dataset.fam").read_text().splitlines()
        assert fam[0].startswith("Yoruba\t"), "forReference prefix stripped"
        metadata = pd.read_csv(raw / "metadata.csv")
        assert list(metadata["project_meta.sample_id"]) == ["S1", "S2"]
        assert list(metadata["Population"]) == ["Yoruba", "French"]

    def test_the_bed_is_renamed_not_rewritten(self, tmp_path):
        """A 5-byte fake .bed: nothing may parse it, only move it."""
        from manifold_genetics.scaffold import _extract_hgdp_archive

        raw = tmp_path / "raw"
        _extract_hgdp_archive(self._workbench_archive(tmp_path), raw)
        assert (raw / "full_dataset.bed").read_bytes() == b"\x6c\x1b\x01\x00\x00"
        assert not (raw / "extractedChrAllUnpruned.bed").exists()

    def test_a_tar_with_a_top_level_directory_still_lands_in_raw(self, tmp_path):
        """The real archive unpacks to 1KGPHGDP/ (prepare_data.sh's REF_DIR)."""
        from manifold_genetics.scaffold import _extract_hgdp_archive

        raw = tmp_path / "raw"
        _extract_hgdp_archive(self._workbench_archive(tmp_path, subdir="1KGPHGDP"), raw)
        assert (raw / "full_dataset.bed").exists()
        assert not (raw / "1KGPHGDP").exists()

    def test_the_log_names_the_layout(self, tmp_path, caplog):
        """So nobody mistakes the unfiltered workbench panel for the public one."""
        import logging

        from manifold_genetics.scaffold import _extract_hgdp_archive

        with caplog.at_level(logging.WARNING, logger="manifold_genetics.scaffold"):
            _extract_hgdp_archive(self._workbench_archive(tmp_path), tmp_path / "raw")
        assert any("orkbench" in r.getMessage() for r in caplog.records)

    def test_a_workbench_archive_fits_on_every_sample(self, tmp_path, monkeypatch):
        """No relatedness metadata, so there is no unrelated subset to pick."""
        from manifold_genetics import scaffold

        keeps, prefixed = {}, {}

        def _fake_keep(bfile, keep, out, plink2, keep_chr_prefix=False):
            keeps[out.name] = keep.read_text()
            prefixed[out.name] = keep_chr_prefix

        monkeypatch.setattr(scaffold, "_run_plink2_keep", _fake_keep)
        out = tmp_path / "out"
        scaffold.acquire_hgdp(out, archive=self._workbench_archive(tmp_path))

        # plink2 --keep matches FID and IID, and the workbench .fam's FID is the
        # population: a keep file of `S1 S1` would select nobody.
        assert keeps["fit_subset"] == keeps["project_subset"] == "Yoruba\tS1\nFrench\tS2\n"
        # And plink2 --make-bed writes chr1 back as 1 unless told otherwise, which
        # would undo the prefix `preprocess --fit-has-chr-prefix` is promised.
        assert prefixed == {"fit_subset": True, "project_subset": True}
        labels = pd.read_csv(out / "data" / "labels.csv")
        assert list(labels.columns) == ["sample_id", "Population"]
        assert list(labels["sample_id"]) == ["S1", "S2"]
        colormap = json.loads((out / "colormap.json").read_text())
        assert set(colormap["Population"]) == {"Yoruba", "French"}
        assert (out / "config.yaml").exists()

    def test_the_public_layout_keeps_numeric_chromosomes(self, tmp_path, monkeypatch):
        """The public .bim is `1`, not `chr1`; nothing must add a prefix there."""
        import shutil
        import tarfile

        from manifold_genetics import scaffold

        src = tmp_path / "public"
        src.mkdir()
        (src / "full_dataset.bed").write_bytes(b"\x6c\x1b\x01\x00\x00")
        (src / "full_dataset.bim").write_text("1\trs1\t0\t100\tA\tG\n22\trs2\t0\t200\tC\tT\n")
        (src / "full_dataset.fam").write_text("S1\tS1\t0\t0\t0\t-9\nS2\tS2\t0\t0\t0\t-9\n")
        (src / "metadata.csv").write_text(
            "project_meta.sample_id,Genetic_region_merged,filter_pca_outlier,"
            "hard_filtered,filter_king_related,filter_contaminated\n"
            "S1,Africa,False,False,False,False\nS2,Europe,False,False,True,False\n"
        )
        archive = tmp_path / "hgdp_1kgp_full.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for f in src.iterdir():
                tar.add(f, arcname=f.name)
        shutil.rmtree(src)

        keeps, prefixed = {}, {}

        def _fake_keep(bfile, keep, out, plink2, keep_chr_prefix=False):
            keeps[out.name] = keep.read_text()
            prefixed[out.name] = keep_chr_prefix

        monkeypatch.setattr(scaffold, "_run_plink2_keep", _fake_keep)
        scaffold.acquire_hgdp(tmp_path / "out", archive=archive)

        assert keeps["fit_subset"] == "S1\tS1\n", "S2 is related"
        assert keeps["project_subset"] == "S1\tS1\nS2\tS2\n"
        assert prefixed == {"fit_subset": False, "project_subset": False}

    def test_plink2_is_told_to_keep_the_chr_prefix_only_when_asked(self, tmp_path, monkeypatch):
        from manifold_genetics import scaffold

        calls = []
        monkeypatch.setattr(scaffold.subprocess, "run", lambda argv, **kw: calls.append(argv))
        keep = tmp_path / "keep.txt"
        keep.write_text("")
        scaffold._run_plink2_keep(tmp_path / "in", keep, tmp_path / "out", "plink2")
        scaffold._run_plink2_keep(
            tmp_path / "in", keep, tmp_path / "out", "plink2", keep_chr_prefix=True
        )
        assert "--output-chr" not in calls[0]
        assert calls[1][calls[1].index("--output-chr") + 1] == "chrM"

    def test_the_config_header_describes_the_panel_it_was_written_for(self, tmp_path, monkeypatch):
        """The public header's "3,400 unrelated" would be a lie above the workbench panel."""
        from manifold_genetics import scaffold

        monkeypatch.setattr(scaffold, "_run_plink2_keep", lambda *a, **k: None)
        out = tmp_path / "out"
        header = scaffold.acquire_hgdp(out, archive=self._workbench_archive(tmp_path)).read_text()

        assert "3,400" not in header
        assert "preprocess --preset harmonise --fit-has-chr-prefix" in header
        assert "fit_plink: data/fit_subset" in header, "only the comment block changes"
        assert "3,400" in scaffold._HGDP_CONFIG, "the public text is unchanged"

    def test_a_rerun_still_knows_the_data_is_the_workbench_panel(self, tmp_path, monkeypatch):
        """Normalising leaves full_dataset.* on disk, which looks public. A second
        run must not therefore go looking for QC columns the metadata lacks."""
        from manifold_genetics import scaffold

        monkeypatch.setattr(scaffold, "_run_plink2_keep", lambda *a, **k: None)
        out = tmp_path / "out"
        archive = self._workbench_archive(tmp_path)
        scaffold.acquire_hgdp(out, archive=archive)
        assert scaffold._detect_hgdp_layout(out / "data" / "raw") == "workbench"

        scaffold.acquire_hgdp(out, archive=archive, force=True)  # no KeyError
        assert list(pd.read_csv(out / "data" / "labels.csv").columns) == ["sample_id", "Population"]

    def test_a_named_archive_that_is_missing_is_an_error_not_a_download(
        self, tmp_path, monkeypatch
    ):
        """Silently fetching the public panel instead would swap the cohort."""
        from manifold_genetics import scaffold

        def _must_not_download(destination):
            raise AssertionError("fell back to downloading the public archive")

        monkeypatch.setattr(scaffold, "_download_hgdp_archive", _must_not_download)
        with pytest.raises(FileNotFoundError, match="nope.tar.gz"):
            scaffold.acquire_hgdp(tmp_path / "out", archive=tmp_path / "nope.tar.gz")

    def test_a_gs_url_is_fetched_with_gsutil(self, tmp_path, monkeypatch):
        from manifold_genetics import scaffold

        calls = []
        monkeypatch.setattr(scaffold.subprocess, "run", lambda argv, **kw: calls.append(argv))
        monkeypatch.setattr(scaffold, "_extract_hgdp_archive", lambda a, r: None)
        monkeypatch.setattr(scaffold, "_run_plink2_keep", lambda *a, **k: None)
        with pytest.raises(FileNotFoundError):  # nothing was really extracted
            scaffold.acquire_hgdp(tmp_path, archive="gs://bucket/1KGPHGDP.tar.gz")
        assert calls and calls[0][:2] == ["gsutil", "cp"]
        assert calls[0][2] == "gs://bucket/1KGPHGDP.tar.gz"
        assert calls[0][3] == str(tmp_path / "data" / "1KGPHGDP.tar.gz")

    def test_a_gs_url_already_fetched_is_not_fetched_again(self, tmp_path, monkeypatch):
        from manifold_genetics import scaffold

        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "1KGPHGDP.tar.gz").write_bytes(b"")
        calls = []
        monkeypatch.setattr(scaffold.subprocess, "run", lambda argv, **kw: calls.append(argv))
        monkeypatch.setattr(scaffold, "_extract_hgdp_archive", lambda a, r: None)
        monkeypatch.setattr(scaffold, "_run_plink2_keep", lambda *a, **k: None)
        with pytest.raises(FileNotFoundError):
            scaffold.acquire_hgdp(tmp_path, archive="gs://bucket/1KGPHGDP.tar.gz")
        assert calls == []


class TestDownloadFallsBackToSystemTools:
    """urllib and curl do not trust the same certificates.

    `examples/hgdp_1kgp/download_data.sh` fetched this archive with wget or
    curl, which use the system certificate store. Porting it to urllib moved the
    trust decision to Python's bundled CA list, which on a network with a
    TLS-intercepting proxy does not include the proxy's root -- so a download
    that had always worked started failing with CERTIFICATE_VERIFY_FAILED.

    Falling back to curl restores the original behaviour. Certificate
    verification stays on in every path: the point is to use the trust store the
    machine's administrators configured, not to skip the check.
    """

    def test_curl_is_used_when_urllib_cannot_verify(self, tmp_path, monkeypatch):
        import urllib.request

        from manifold_genetics import scaffold

        def _tls_failure(url, dest):
            raise OSError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate")

        calls = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"archive")
            return None

        from manifold_genetics.utils import tools

        monkeypatch.setattr(urllib.request, "urlretrieve", _tls_failure)
        monkeypatch.setattr(tools.shutil, "which", lambda n: f"/usr/bin/{n}")
        monkeypatch.setattr(tools.subprocess, "run", _fake_run)

        archive = scaffold._download_hgdp_archive(tmp_path)

        assert archive.exists()
        assert calls and calls[0][0].endswith("curl"), f"expected curl, got {calls}"

    def test_certificate_verification_is_never_disabled(self):
        """A download that skips verification is worse than one that fails."""
        from manifold_genetics import scaffold
        from manifold_genetics.utils import tools

        source = Path(scaffold.__file__).read_text() + Path(tools.__file__).read_text()

        for forbidden in ("--insecure", "-k ", "verify=False", "_create_unverified_context"):
            assert forbidden not in source, f"{forbidden!r} disables certificate checking"


class TestHgdpLabels:
    """The labels have to actually label something.

    `acquire hgdp` looked for a column `project_meta.genetic_region`, which the
    cohort's metadata does not have -- it is `Genetic_region_merged` -- and fell
    back to writing "Unknown" for every sample. The run completed, the figure was
    drawn, and every point in it was the same grey. Reported 2026-09-13.

    A label file that silently does not match its cohort is this project's most
    expensive failure mode; it has caused published figures to colour 40% of
    their points from a superseded selection. So the fallback is gone: a missing
    column is now an error.
    """

    @pytest.fixture
    def metadata(self):
        return pd.DataFrame(
            {
                "project_meta.sample_id": ["a", "b", "c"],
                "Genetic_region_merged": ["Africa", "Europe", "East_Asia"],
                "filter_pca_outlier": [False] * 3,
                "hard_filtered": [False] * 3,
                "filter_king_related": [False] * 3,
                "filter_contaminated": [False] * 3,
            }
        )

    def test_labels_carry_the_real_regions(self, tmp_path, metadata):
        from manifold_genetics.scaffold import _write_hgdp_labels

        _write_hgdp_labels(
            metadata,
            metadata["project_meta.sample_id"],
            tmp_path / "labels.csv",
            tmp_path / "colormap.json",
        )

        labels = pd.read_csv(tmp_path / "labels.csv")
        assert sorted(labels.iloc[:, 1]) == ["Africa", "East_Asia", "Europe"]
        assert "Unknown" not in set(labels.iloc[:, 1])

    def test_the_colormap_uses_the_published_colours(self, tmp_path, metadata):
        """So a figure from `acquire hgdp` is comparable with the shipped example's."""
        from manifold_genetics.scaffold import _write_hgdp_labels

        _write_hgdp_labels(
            metadata,
            metadata["project_meta.sample_id"],
            tmp_path / "labels.csv",
            tmp_path / "colormap.json",
        )

        colours = json.loads((tmp_path / "colormap.json").read_text())
        column = next(iter(colours))
        assert colours[column]["Africa"] == "#008000"
        assert colours[column]["East_Asia"] == "#0000FF"

    def test_a_missing_region_column_is_an_error_not_a_placeholder(self, tmp_path, metadata):
        from manifold_genetics.scaffold import _write_hgdp_labels

        without = metadata.drop(columns=["Genetic_region_merged"])

        with pytest.raises(KeyError, match="Genetic_region_merged"):
            _write_hgdp_labels(
                without,
                without["project_meta.sample_id"],
                tmp_path / "labels.csv",
                tmp_path / "colormap.json",
            )


class TestInitCustom:
    """Scaffolding a config for genotypes the user already has.

    The YAML is fifteen obvious lines; the colormap is not. UK Biobank's needs
    22 hex colours for `self_described_ancestry` alone, hand-written into JSON,
    and a value missing from it is drawn grey and dropped from the legend. The
    other trap is sample IDs: a label file that does not match the .fam does not
    crash, it colours a fraction of the points.

    So this writes the colormap and checks the overlap, which are the two things
    a person should not be doing by hand.
    """

    @pytest.fixture
    def cohort(self, tmp_path):
        from manifold_genetics.scaffold import write_bed

        rng = __import__("numpy").random.default_rng(0)
        dosages = rng.binomial(2, 0.3, size=(40, 60)).astype("uint8")
        ids = [f"S{i:03d}" for i in range(40)]
        write_bed(tmp_path / "cohort", dosages, ids)
        pd.DataFrame({"sample_id": ids, "population": ["A", "B", "C", "D"] * 10}).to_csv(
            tmp_path / "labels.csv", index=False
        )
        return tmp_path

    def test_writes_a_config_the_loader_accepts(self, cohort):
        from manifold_genetics.pipeline.configfile import load_config
        from manifold_genetics.scaffold import acquire_custom

        config = acquire_custom(
            cohort / "out", fit_plink=cohort / "cohort", labels=cohort / "labels.csv"
        )

        kwargs = load_config(config)
        assert kwargs["fit_plink"] == cohort / "cohort"

    def test_generates_a_colour_for_every_label_value(self, cohort):
        """The 22-colours-by-hand problem is the reason this command exists."""
        from manifold_genetics.scaffold import acquire_custom

        acquire_custom(cohort / "out", fit_plink=cohort / "cohort", labels=cohort / "labels.csv")

        colours = json.loads((cohort / "out" / "colormap.json").read_text())
        assert set(colours["population"]) == {"A", "B", "C", "D"}
        assert all(c.startswith("#") for c in colours["population"].values())

    def test_refuses_labels_that_do_not_match_the_genotypes(self, cohort):
        """The silent failure: a stale label file colours a fraction of the points."""
        from manifold_genetics.scaffold import acquire_custom

        pd.DataFrame(
            {"sample_id": [f"OTHER{i}" for i in range(40)], "population": ["A"] * 40}
        ).to_csv(cohort / "wrong.csv", index=False)

        with pytest.raises(ValueError, match="different datasets"):
            acquire_custom(cohort / "out", fit_plink=cohort / "cohort", labels=cohort / "wrong.csv")

    def test_project_plink_defaults_to_the_fit_set(self, cohort):
        """The common case is one cohort, embedded whole."""
        from manifold_genetics.pipeline.configfile import load_config
        from manifold_genetics.scaffold import acquire_custom

        config = acquire_custom(
            cohort / "out", fit_plink=cohort / "cohort", labels=cohort / "labels.csv"
        )

        kwargs = load_config(config)
        assert kwargs["project_plink"] == kwargs["fit_plink"]

    def test_missing_plink_files_are_reported_before_anything_is_written(self, cohort):
        from manifold_genetics.scaffold import acquire_custom

        with pytest.raises(FileNotFoundError, match="bim"):
            acquire_custom(
                cohort / "out", fit_plink=cohort / "absent", labels=cohort / "labels.csv"
            )
        assert not (cohort / "out" / "config.yaml").exists()


class TestInitAou:
    """All of Us is controlled-access and lives in a workbench.

    Its data comes from `gs://fc-aou-datasets-controlled` and needs
    GOOGLE_PROJECT, gsutil and bq, which only exist inside the Researcher
    Workbench. What the command does once it is there -- the download, the FAM
    fix, the BigQuery metadata, the labels -- is tested against stubs in
    `test_aou_acquire.py`. The failure path is what almost everyone who types
    this will see, so it is the part that is tested here.
    """

    def test_outside_the_workbench_it_says_so_and_writes_nothing(self, tmp_path, monkeypatch):
        from manifold_genetics import scaffold

        monkeypatch.delenv("GOOGLE_PROJECT", raising=False)
        monkeypatch.setattr(scaffold.shutil, "which", lambda n: None)

        with pytest.raises(EnvironmentError) as excinfo:
            scaffold.acquire_aou(tmp_path)

        message = str(excinfo.value)
        assert "GOOGLE_PROJECT" in message
        assert "Researcher Workbench" in message
        assert not (tmp_path / "config.yaml").exists(), "nothing should be written on refusal"

    def test_it_names_every_missing_prerequisite_at_once(self, tmp_path, monkeypatch):
        """Reporting them one per run would be three round trips."""
        from manifold_genetics import scaffold

        monkeypatch.delenv("GOOGLE_PROJECT", raising=False)
        monkeypatch.setattr(scaffold.shutil, "which", lambda n: None)

        with pytest.raises(EnvironmentError) as excinfo:
            scaffold.acquire_aou(tmp_path)

        for expected in ("GOOGLE_PROJECT", "gsutil", "bq"):
            assert expected in str(excinfo.value)

    def test_plink_is_not_a_prerequisite(self):
        """The script's per-chromosome split was the only plink call, and it
        is not ported: nothing downstream read the per-chromosome files."""
        from manifold_genetics import scaffold

        assert "plink2" not in scaffold._AOU_REQUIRED_TOOLS
        assert "plink" not in scaffold._AOU_REQUIRED_TOOLS

    def test_the_config_is_the_cohort_alone(self):
        """`acquire aou` writes the cohort as a whole_cohort directory;
        examples/aou/hgdp_1kgp_proj/config.yaml is the *projection* of it onto
        HGDP+1KGP, which `preprocess` now produces from two cohort directories.
        So the two are no longer compared -- only what carries over is pinned:
        the PCA count and the batch size that bounds memory on 400k+ samples.
        """
        import yaml

        from manifold_genetics import scaffold

        written = yaml.safe_load(scaffold._AOU_CONFIG)

        assert written["preset"] == "whole_cohort"
        assert written["data"]["fit_plink"] == written["data"]["project_plink"]
        assert written["pca"]["n_pcs"] == 20
        assert written["embedding"]["embed_batch_size"] == 60000
        assert written["skip"]["admixture"] is True
        assert "visualization" not in written, "projection_plot_* are preprocess's to set"


class TestInitCustomSeparateLabels:
    """Real cohorts often label the two sets separately.

    Every UK Biobank config uses `fit_labels` and `project_labels`, because the
    fit subset and the full cohort are described by different files. With only
    `--labels` the command could not express the configs this package ships,
    so the documented example had to pretend otherwise.
    """

    @pytest.fixture
    def cohort(self, tmp_path):
        from manifold_genetics.scaffold import write_bed

        rng = __import__("numpy").random.default_rng(0)
        ids = [f"S{i:03d}" for i in range(40)]
        write_bed(tmp_path / "fit", rng.binomial(2, 0.3, size=(40, 60)).astype("uint8"), ids)
        write_bed(tmp_path / "proj", rng.binomial(2, 0.3, size=(40, 60)).astype("uint8"), ids)
        for name in ("fit_labels", "project_labels"):
            pd.DataFrame({"sample_id": ids, "group": ["A", "B"] * 20}).to_csv(
                tmp_path / f"{name}.csv", index=False
            )
        return tmp_path

    def test_separate_label_files_reach_the_config(self, cohort):
        from manifold_genetics.pipeline.configfile import load_config
        from manifold_genetics.scaffold import acquire_custom

        config = acquire_custom(
            cohort / "out",
            fit_plink=cohort / "fit",
            project_plink=cohort / "proj",
            fit_labels=cohort / "fit_labels.csv",
            project_labels=cohort / "project_labels.csv",
            preset="subsample",
        )

        kwargs = load_config(config)
        assert kwargs["fit_labels"] == cohort / "fit_labels.csv"
        assert kwargs["project_labels"] == cohort / "project_labels.csv"

    def test_one_label_file_still_works(self, cohort):
        from manifold_genetics.pipeline.configfile import load_config
        from manifold_genetics.scaffold import acquire_custom

        config = acquire_custom(
            cohort / "out", fit_plink=cohort / "fit", labels=cohort / "fit_labels.csv"
        )

        assert load_config(config)["labels"] == cohort / "fit_labels.csv"

    def test_it_asks_for_labels_of_some_kind(self, cohort):
        from manifold_genetics.scaffold import acquire_custom

        with pytest.raises(ValueError, match="labels"):
            acquire_custom(cohort / "out", fit_plink=cohort / "fit")
