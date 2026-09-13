"""Scaffolding for a fresh install: a runnable config, and data to run it on.

``pip install manifold-genetics`` ships no example config and no data --
``examples/`` is in neither the wheel nor the sdist. So the documented first
command, ``manifold-genetics run config.yaml``, had nothing to run: there was no
config to point it at and no template to copy one from. This module is what
``manifold-genetics init`` uses to fix that.

Two targets, because they answer different questions:

- ``synthetic`` simulates a small cohort and writes everything beside it. No
  network, a few seconds, and the result runs end to end -- which makes it the
  way to confirm an installation actually works.
- ``hgdp`` fetches the real HGDP+1KGP cohort and prepares it exactly as
  ``examples/hgdp_1kgp/`` does, so what you get is the cohort the published
  figures were made from.
"""

import logging
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# The same archive examples/hgdp_1kgp/download_data.sh fetches: full_dataset
# .bed/.bim/.fam (4,151 samples, 172,152 SNPs) plus metadata.csv.
HGDP_ARCHIVE_URL = (
    "https://www.dropbox.com/scl/fi/f8kay7kgpgb2s5ncsykpg/hgdp_1kgp_full.tar.gz"
    "?rlkey=pp7fj711zxlg4o2lbskxj5rwe&st=jv1xh8vo&dl=1"
)

# PLINK 1 .bed codes, by A1 dosage. 01 is missing and is never written here.
_DOSAGE_TO_CODE = {2: 0b00, 1: 0b10, 0: 0b11}

# Deliberately not real population or region names. These figures are of
# fabricated genotypes, and a plot labelled with real ancestries can be
# mistaken for a real result once it is separated from the command that
# made it.
_SYNTHETIC_GROUPS = ("Group A", "Group B", "Group C")
_SYNTHETIC_COLOURS = ("#4292C6", "#C7E9C0", "#E3242B")


def _refuse_to_clobber(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. Pass --force to overwrite it, or --out to write "
            "somewhere else."
        )


def write_bed(prefix: Path, dosages: np.ndarray, sample_ids: Sequence[str]) -> None:
    """Write a PLINK 1 ``.bed``/``.bim``/``.fam`` triple from A1 dosages.

    SNP-major, two bits per sample, which is what ``read_bed_dosages`` expects.
    Written here rather than shelled out to plink2, because the whole point of
    the synthetic target is that it needs no external binary.

    Args:
        prefix: Path prefix; the three extensions are appended.
        dosages: ``(n_samples, n_variants)`` integer A1 dosages in {0, 1, 2}.
        sample_ids: One identifier per sample, used as both FID and IID.
    """
    n_samples, n_variants = dosages.shape
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{prefix}.bed", "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))
        for variant in range(n_variants):
            row = bytearray()
            for start in range(0, n_samples, 4):
                byte = 0
                for offset, sample in enumerate(range(start, min(start + 4, n_samples))):
                    byte |= _DOSAGE_TO_CODE[int(dosages[sample, variant])] << (2 * offset)
                row.append(byte)
            fh.write(bytes(row))

    Path(f"{prefix}.fam").write_text("".join(f"{s} {s} 0 0 0 -9\n" for s in sample_ids))
    Path(f"{prefix}.bim").write_text("".join(f"1 rs{v} 0 {v + 1} A G\n" for v in range(n_variants)))


def simulate_cohort(
    n_samples: int = 240,
    n_variants: int = 800,
    seed: int = 0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """A small cohort with real population structure.

    Three groups with differing allele frequencies, so PCA and the embedding
    have something to find -- a cohort of noise would run but would show
    nothing, and the point of the synthetic target is to demonstrate the
    pipeline working.

    Returns:
        ``(dosages, labels)`` -- an integer ``(n_samples, n_variants)`` array and
        a frame with ``sample_id`` and ``population``.
    """
    rng = np.random.default_rng(seed)
    n_groups = len(_SYNTHETIC_GROUPS)
    assignment = np.arange(n_samples) % n_groups

    # Each group gets its own frequency profile over a shared variant set.
    frequencies = rng.uniform(0.1, 0.9, size=(n_groups, n_variants))
    dosages = np.empty((n_samples, n_variants), dtype=np.uint8)
    for sample in range(n_samples):
        dosages[sample] = rng.binomial(2, frequencies[assignment[sample]])

    labels = pd.DataFrame(
        {
            "sample_id": [f"SIM{i:04d}" for i in range(n_samples)],
            "population": [_SYNTHETIC_GROUPS[g] for g in assignment],
        }
    )
    return dosages, labels


_SYNTHETIC_CONFIG = """\
# Written by `manifold-genetics init synthetic`.
#
# A simulated cohort of {n_samples} samples across {n_groups} populations, with
# real structure to find. Nothing here needs downloading, and the whole run takes
# under a minute -- it is the quickest way to confirm an installation works.
#
#   manifold-genetics run config.yaml --dry-run   # print the settings, do nothing
#   manifold-genetics run config.yaml             # do the work
#
# Paths are relative to this file.

# Which cohort the model is fitted on is the choice that defines an analysis.
# `whole_cohort` fits on the fit set and embeds the whole project set. The other
# presets are `projection` (fit a reference panel, project your cohort onto it)
# and `subsample` (embed a subset of a cohort too large to embed whole).
preset: whole_cohort

data:
  fit_plink: data/fit_subset        # PLINK prefix -- .bed/.bim/.fam
  project_plink: data/project_subset
  labels: data/labels.csv           # sample_id, plus any column to colour by
  colormap: colormap.json           # which columns to plot, and in what colours
  output_dir: outputs

pca:
  n_pcs: 10

embedding:
  method: phate

# Admixture needs the `admixture` extra (torch), so it is off by default here.
skip:
  admixture: true
"""


def init_synthetic(out_dir: PathLike, force: bool = False, seed: int = 0) -> Path:
    """Write a simulated cohort and a config that runs on it.

    Args:
        out_dir: Directory to write into; created if absent.
        force: Overwrite an existing ``config.yaml`` rather than refusing.
        seed: Seed for the simulation, so the cohort is reproducible.

    Returns:
        The path of the config file written.

    Raises:
        FileExistsError: ``config.yaml`` exists and ``force`` is False.
    """
    out_dir = Path(out_dir)
    config_path = out_dir / "config.yaml"
    _refuse_to_clobber(config_path, force)

    dosages, labels = simulate_cohort(seed=seed)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    # The fit set is half the cohort; the project set is all of it, which is the
    # shape `whole_cohort` describes.
    fit = np.arange(0, len(labels), 2)
    write_bed(out_dir / "data" / "fit_subset", dosages[fit], labels["sample_id"].iloc[fit])
    write_bed(out_dir / "data" / "project_subset", dosages, labels["sample_id"])

    labels.to_csv(out_dir / "data" / "labels.csv", index=False)
    colours = dict(zip(_SYNTHETIC_GROUPS, _SYNTHETIC_COLOURS))
    (out_dir / "colormap.json").write_text(
        '{\n  "population": {\n'
        + ",\n".join(f'    "{g}": "{c}"' for g, c in colours.items())
        + "\n  }\n}\n"
    )
    config_path.write_text(
        _SYNTHETIC_CONFIG.format(n_samples=len(labels), n_groups=len(_SYNTHETIC_GROUPS))
    )

    logger.info("Wrote a simulated cohort and config to %s", out_dir)
    return config_path


def hgdp_subsets(metadata: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """The fit and project sample sets, from the cohort's metadata.

    Ported from ``examples/hgdp_1kgp/prepare_data.sh``, which produced the
    subsets every published figure was made from. Both drop PCA outliers,
    hard-filtered and contaminated samples; the fit set additionally drops
    related samples, because a model fitted on relatives is fitted partly on
    the same genome twice.

    Returns:
        ``(fit_ids, project_ids)`` -- 3,400 and 4,094 samples on the real cohort.
    """
    passes_qc = (
        ~metadata["filter_pca_outlier"].astype(bool)
        & ~metadata["hard_filtered"].astype(bool)
        & ~metadata["filter_contaminated"].astype(bool)
    )
    unrelated = ~metadata["filter_king_related"].astype(bool)

    ids = metadata["project_meta.sample_id"]
    return ids[passes_qc & unrelated], ids[passes_qc]


def _fetch_with_system_tool(url: str, archive: Path) -> bool:
    """Download with curl or wget, returning whether it worked.

    These use the operating system's certificate store rather than Python's
    bundled one. On a network with a TLS-intercepting proxy the system store
    usually already trusts the proxy's root, because an administrator put it
    there, while Python's does not -- so this succeeds where urllib raises
    CERTIFICATE_VERIFY_FAILED. It is also what the shell script this was ported
    from always did.

    Verification is not disabled in either path. The trust decision is delegated
    to the machine's configuration, not skipped.
    """
    for tool, command in (
        ("curl", ["curl", "-fsSL", url, "-o", str(archive)]),
        ("wget", ["wget", "-q", url, "-O", str(archive)]),
    ):
        executable = shutil.which(tool)
        if not executable:
            continue
        logger.info("Downloading with %s", tool)
        try:
            subprocess.run([executable] + command[1:], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.info("%s failed: %s", tool, exc)
            archive.unlink(missing_ok=True)
            continue
        if archive.exists() and archive.stat().st_size > 0:
            return True
    return False


def _download_hgdp_archive(destination: Path) -> Path:
    archive = destination / "hgdp_1kgp_full.tar.gz"
    if archive.exists():
        logger.info("Archive already present: %s", archive)
        return archive

    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HGDP+1KGP (about 183 MB) to %s", archive)
    try:
        # Do NOT make this the only way this file is fetched. The shell script
        # this was ported from (examples/hgdp_1kgp/download_data.sh) used wget or
        # curl, which read the *operating system's* certificate store. urllib
        # reads Python's bundled one instead, and on a network with a
        # TLS-intercepting proxy that bundle does not contain the proxy's root
        # certificate -- so this call raises CERTIFICATE_VERIFY_FAILED on
        # machines where the download had always worked. Reported 2026-09-13,
        # within hours of the port shipping. The fallback below is what restores
        # the original behaviour; removing it reintroduces that regression.
        urllib.request.urlretrieve(HGDP_ARCHIVE_URL, archive)
        return archive
    except OSError as exc:
        logger.info("urllib could not fetch it (%s); trying curl or wget", exc)
        if _fetch_with_system_tool(HGDP_ARCHIVE_URL, archive):
            return archive
        # Most often a TLS-intercepting proxy, whose root certificate Python
        # does not trust, or a host with no route out at all -- an HPC compute
        # node, say. Neither is fixable from here, but both are worked around
        # the same way, so the error has to say how rather than suggest a retry.
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the HGDP+1KGP archive: {exc}\n\n"
            "If this is a proxy or an offline machine, fetch it by other means and "
            "point init at the file:\n\n"
            f"    curl -L -o hgdp_1kgp_full.tar.gz '{HGDP_ARCHIVE_URL}'\n"
            "    manifold-genetics init hgdp --archive hgdp_1kgp_full.tar.gz\n\n"
            "curl and wget use the system certificate store, which on a managed "
            "network usually already trusts the proxy."
        ) from exc
    return archive


def _extract_hgdp_archive(archive: Path, raw_dir: Path) -> None:
    if (raw_dir / "full_dataset.bed").exists():
        logger.info("Raw data already extracted in %s", raw_dir)
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s", archive.name)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            # Flatten, and never write outside the target directory.
            name = Path(member.name).name
            if not name or member.isdir():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            (raw_dir / name).write_bytes(extracted.read())


_HGDP_CONFIG = """\
# Written by `manifold-genetics init hgdp`.
#
# HGDP+1KGP: 4,094 QC-passing samples across seven genetic regions, with the
# model fitted on the 3,400 that are also unrelated. This is the cohort the
# published figures were made from.
#
#   manifold-genetics run config.yaml --dry-run   # print the settings, do nothing
#   manifold-genetics run config.yaml             # do the work
#
# Paths are relative to this file.

preset: whole_cohort

data:
  fit_plink: data/fit_subset
  project_plink: data/project_subset
  labels: data/labels.csv
  colormap: colormap.json
  output_dir: outputs

pca:
  n_pcs: 20

embedding:
  method: phate

# Admixture needs the `admixture` extra (torch), so it is off by default here.
skip:
  admixture: true
"""


def init_hgdp(
    out_dir: PathLike,
    force: bool = False,
    download: bool = True,
    plink2: Optional[str] = None,
    archive: Optional[PathLike] = None,
) -> Path:
    """Fetch and prepare HGDP+1KGP, and write a config that runs on it.

    Downloads about 183 MB, then subsets it with plink2 exactly as
    ``examples/hgdp_1kgp/prepare_data.sh`` does. plink2 is resolved through the
    usual chain and fetched if missing, so this needs internet -- on a cluster,
    run it on a login node.

    Args:
        out_dir: Directory to write into.
        force: Overwrite an existing ``config.yaml``.
        download: Fetch the archive if no local copy is found. False still
            extracts one that is already present.
        plink2: Path to plink2; resolved automatically when None.
        archive: An already-downloaded ``hgdp_1kgp_full.tar.gz`` to use instead
            of fetching. For proxied networks and offline machines.

    Returns:
        The path of the config file written.
    """
    out_dir = Path(out_dir)
    config_path = out_dir / "config.yaml"
    _refuse_to_clobber(config_path, force)

    data_dir = out_dir / "data"
    raw_dir = data_dir / "raw"

    if not (raw_dir / "full_dataset.bed").exists():
        # Prefer a file we already have over the network, in this order: one the
        # caller named, one left in place by an earlier run, then downloading.
        # `download=False` means "do not fetch", not "do not unpack" -- someone
        # who obtained the archive another way still needs it extracted.
        local = Path(archive) if archive else data_dir / "hgdp_1kgp_full.tar.gz"
        if local.exists():
            _extract_hgdp_archive(local, raw_dir)
        elif download:
            _extract_hgdp_archive(_download_hgdp_archive(data_dir), raw_dir)

    if not (raw_dir / "full_dataset.bed").exists():
        raise FileNotFoundError(
            f"{raw_dir}/full_dataset.bed is missing, and no archive was found at "
            f"{data_dir}/hgdp_1kgp_full.tar.gz. Download it and pass --archive, or "
            "drop --no-download to fetch it."
        )

    metadata = pd.read_csv(raw_dir / "metadata.csv")
    fit_ids, project_ids = hgdp_subsets(metadata)
    logger.info("Selected %d fit and %d project samples", len(fit_ids), len(project_ids))

    for name, ids in (("fit", fit_ids), ("project", project_ids)):
        keep = data_dir / f"{name}_indices.txt"
        keep.write_text("".join(f"{i}\t{i}\n" for i in ids))
        _run_plink2_keep(raw_dir / "full_dataset", keep, data_dir / f"{name}_subset", plink2)

    _write_hgdp_labels(metadata, project_ids, data_dir / "labels.csv", out_dir / "colormap.json")
    config_path.write_text(_HGDP_CONFIG)

    logger.info("Wrote HGDP+1KGP and a config to %s", out_dir)
    return config_path


def _run_plink2_keep(bfile: Path, keep: Path, out: Path, plink2: Optional[str]) -> None:
    import subprocess

    if plink2 is None:
        from .utils.tools import ToolResolver

        plink2 = ToolResolver().resolve_plink2()

    logger.info("Creating %s with plink2", out.name)
    subprocess.run(
        [
            plink2,
            "--bfile",
            str(bfile),
            "--keep",
            str(keep),
            "--make-bed",
            "--out",
            str(out),
            "--silent",
        ],
        check=True,
    )


# The metadata column naming the genetic region each sample belongs to, in the
# order the published figures colour them.
_HGDP_REGION_COLUMN = "project_meta.genetic_region"
_HGDP_REGION_COLOURS = {
    "AFR": "#008000",
    "AMR": "#FF0000",
    "CSA": "#FFA500",
    "EAS": "#0000FF",
    "EUR": "#9370DB",
    "MID": "#8B4513",
    "OCE": "#FF69B4",
}


def _write_hgdp_labels(
    metadata: pd.DataFrame, project_ids: pd.Series, labels_path: Path, colormap_path: Path
) -> None:
    keep = metadata[metadata["project_meta.sample_id"].isin(set(project_ids))]
    region = (
        keep[_HGDP_REGION_COLUMN]
        if _HGDP_REGION_COLUMN in keep.columns
        else pd.Series(["Unknown"] * len(keep), index=keep.index)
    )
    pd.DataFrame({"sample_id": keep["project_meta.sample_id"], "genetic_region": region}).to_csv(
        labels_path, index=False
    )

    present = sorted(set(region.astype(str)))
    colours = {r: _HGDP_REGION_COLOURS.get(r, "#999999") for r in present}
    colormap_path.write_text(
        '{\n  "genetic_region": {\n'
        + ",\n".join(f'    "{r}": "{c}"' for r, c in colours.items())
        + "\n  }\n}\n"
    )


def clean(out_dir: PathLike) -> None:
    """Remove a scaffolded directory. Used by tests, not by the CLI."""
    shutil.rmtree(Path(out_dir), ignore_errors=True)
