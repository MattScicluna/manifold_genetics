"""Scaffolding for a fresh install: a runnable config, and data to run it on.

``pip install manifold-genetics`` ships no example config and no data --
``examples/`` is in neither the wheel nor the sdist. So the documented first
command, ``manifold-genetics run config.yaml``, had nothing to run: there was no
config to point it at and no template to copy one from. This module is what
``manifold-genetics acquire`` uses to fix that.

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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# All of Us lives in its own module: the port of download_aou_data.sh is long
# enough to deserve one. Re-exported here because `acquire` has always found it
# under this name, and so have the tests. aou imports this module's helpers
# lazily, so this is not circular.
from .aou import _AOU_CONFIG  # noqa: F401
from .aou import _AOU_REQUIRED_ENV  # noqa: F401
from .aou import _AOU_REQUIRED_TOOLS  # noqa: F401
from .aou import _run_gsutil, acquire_aou, aou_environment_problems  # noqa: F401

# HGDP lives in hgdp.py; re-exported so existing imports and tests keep working.
# hgdp imports this module's plink helpers lazily, so this is not circular.
from .hgdp import _HGDP_CONFIG  # noqa: F401
from .hgdp import _HGDP_CONFIG_BODY  # noqa: F401
from .hgdp import _HGDP_GEOGRAPHY_EXCLUDED_POPULATIONS  # noqa: F401
from .hgdp import _HGDP_POPULATION_COLOURS  # noqa: F401
from .hgdp import _HGDP_POPULATION_COLUMN  # noqa: F401
from .hgdp import _HGDP_PUBLIC_HEADER  # noqa: F401
from .hgdp import _HGDP_REGION_COLOURS  # noqa: F401
from .hgdp import _HGDP_REGION_COLUMN  # noqa: F401
from .hgdp import _HGDP_WORKBENCH_HEADER  # noqa: F401
from .hgdp import _PUBLIC_PREFIX  # noqa: F401
from .hgdp import _WORKBENCH_FID_PREFIX  # noqa: F401
from .hgdp import _WORKBENCH_PREFIX  # noqa: F401
from .hgdp import HGDP_ARCHIVE_URL  # noqa: F401
from .hgdp import _detect_hgdp_layout  # noqa: F401
from .hgdp import _download_hgdp_archive  # noqa: F401
from .hgdp import _extract_hgdp_archive  # noqa: F401
from .hgdp import _hgdp_config  # noqa: F401
from .hgdp import _normalise_workbench_layout  # noqa: F401
from .hgdp import _write_hgdp_geographic  # noqa: F401
from .hgdp import _write_hgdp_labels  # noqa: F401
from .hgdp import acquire_hgdp, hgdp_subsets  # noqa: F401
from .preprocessing.cohort import MIN_LABEL_COVERAGE

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# PLINK 1 .bed codes, by A1 dosage. 01 is missing and is never written here.
_DOSAGE_TO_CODE = {2: 0b00, 1: 0b10, 0: 0b11}

# Deliberately not real population names. These figures are of fabricated
# genotypes, and a plot labelled with real ancestries can be mistaken for a
# real result once it is separated from the command that made it.
_BRANCH_LABEL = "Branch {}"

# The colours manylatents draws this tree with, so a figure made here reads the
# same as one made there.
_BRANCH_COLOURS = {
    1: "#e41a1c",  # red
    2: "#377eb8",  # blue
    3: "#4daf4a",  # green
    4: "#984ea3",  # purple
    5: "#ff7f00",  # orange
    6: "#ffff33",  # yellow
    7: "#a65628",  # brown
    8: "#f781bf",  # pink
}

# The tree the synthetic cohort lies along: ``(from_node, to_node, edge_id,
# n_samples)``, in generation order. This is ``dla_tree_from_graph.yaml`` from
# manylatents, copied rather than imported because that package brings torch
# and lightning with it. Every edge is a branch of samples; the four in
# ``DLA_TREE_GAPS`` are walked, so that what lies beyond them starts in the
# right place, and then dropped, which leaves the cohort in disconnected pieces
# the way a real one with unsampled populations would be.
#
#   N1 -1- N2 -2- N3          gaps (dashed):   N2 ~12~ N8
#           |\_6_ N10 ~11~ N11 -7- N12          N13 ~9~ N4
#            \~12~ N8 -3- N13 ~9~ N4 -4- N5     N5 ~10~ N6
#                              ~10~ N6 -5- N7   N10 ~11~ N11
#                                    \_8_ N9
DLA_TREE_EDGES = (
    (1, 2, 1, 300),  # main trunk
    (2, 3, 2, 300),
    (8, 13, 3, 300),
    (4, 5, 4, 300),
    (6, 7, 5, 100),
    (2, 10, 6, 300),  # branch A, from N2
    (11, 12, 7, 100),
    (6, 9, 8, 300),  # branch B, from N6
    (13, 4, 9, 50),  # gaps
    (5, 6, 10, 50),
    (10, 11, 11, 50),
    (2, 8, 12, 50),
)
DLA_TREE_GAPS = (9, 10, 11, 12)

# Drawn beside the config by `acquire synthetic`, so the shape the embedding is
# supposed to recover is on disk next to the embedding.
GROUND_TRUTH_FIGURE = "dla_tree_ground_truth.png"


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


def dla_tree(
    n_dim: int = 100,
    rand_multiplier: float = 2.0,
    sigma: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Points along :data:`DLA_TREE_EDGES`, by diffusion-limited aggregation.

    Each edge is a random walk that starts where its parent edge ended. Where
    several edges leave one node, each walks in its own block of dimensions and
    holds the rest fixed, so sibling branches are orthogonal by construction
    rather than by luck. Gap edges are walked and then dropped.

    A port of manylatents' ``DLATreeFromGraph``, random draw for random draw:
    the same seed gives the same array there and here, which is what makes a
    figure from this package comparable with one from that. That is also why
    it uses the legacy ``RandomState`` -- it is what the original seeds.

    Args:
        n_dim: Dimensionality of the walk.
        rand_multiplier: Width of each step's uniform increment.
        sigma: Standard deviation of the Gaussian noise added at the end.
        seed: Seed for the walk and the noise.

    Returns:
        ``(coordinates, branch)`` -- a float ``(n_samples, n_dim)`` array and an
        int vector of which data edge each row lies on, numbered 1 to 8.
    """
    rng = np.random.RandomState(seed)

    to_nodes = {to_node for _, to_node, _, _ in DLA_TREE_EDGES}
    nodes = {node for edge in DLA_TREE_EDGES for node in edge[:2]}
    root = next(node for node in sorted(nodes) if node not in to_nodes)

    # Sibling edges split the dimensions between them, in edge order, the
    # first siblings taking the remainder.
    outgoing: Dict[int, List[int]] = {}
    for from_node, _, edge_id, _ in DLA_TREE_EDGES:
        outgoing.setdefault(from_node, []).append(edge_id)
    allowed = {}
    for edge_ids in outgoing.values():
        if len(edge_ids) > 1:
            for edge_id, dims in zip(edge_ids, np.array_split(np.arange(n_dim), len(edge_ids))):
                allowed[edge_id] = dims

    # Walk each edge once its start is known. Edges are listed trunk-first
    # but not in dependency order, so this takes a few passes; the order in
    # which they get walked is what fixes the random sequence.
    position = {root: np.zeros(n_dim)}
    walks: Dict[int, np.ndarray] = {}
    while len(walks) < len(DLA_TREE_EDGES):
        walked_before = len(walks)
        for from_node, to_node, edge_id, length in DLA_TREE_EDGES:
            if edge_id in walks or from_node not in position:
                continue
            dims = allowed.get(edge_id, np.arange(n_dim))
            steps = -0.5 + rand_multiplier * rng.rand(length, len(dims))
            steps[0] = 0  # the first sample sits exactly on the node
            walk = np.tile(position[from_node], (length, 1))
            walk[:, dims] += 0.3 * np.cumsum(steps, axis=0)
            walks[edge_id] = walk
            position[to_node] = walk[-1].copy()
        if len(walks) == walked_before:
            raise ValueError("DLA_TREE_EDGES has an edge that no path from the root reaches")

    coordinates = np.vstack([walks[edge_id] for _, _, edge_id, _ in DLA_TREE_EDGES])
    edge_of_row = np.concatenate([np.full(n, edge_id) for _, _, edge_id, n in DLA_TREE_EDGES])
    if sigma > 0:
        coordinates += rng.normal(0, sigma, coordinates.shape)

    keep = ~np.isin(edge_of_row, DLA_TREE_GAPS)
    _, branch = np.unique(edge_of_row[keep], return_inverse=True)
    return coordinates[keep], branch + 1


def _tree_layout(edges: Sequence[Tuple]) -> Dict[int, Tuple[float, float]]:
    """Node positions for drawing the tree: root on top, one row per depth.

    The same rule as manylatents' ``DLATreeGraphVisualizer``, so the two
    packages draw the same picture of the same tree: children go one row
    below their parent, an only child directly beneath it and siblings spread
    three units apart around it, in the order their edges are listed.
    """
    children: Dict[int, List[int]] = {}
    for from_node, to_node, _, _ in edges:
        children.setdefault(from_node, []).append(to_node)
    to_nodes = {to_node for _, to_node, _, _ in edges}
    root = min(node for node in children if node not in to_nodes)

    rows = [[root]]
    parent_of: Dict[int, int] = {}
    seen = {root}
    while rows[-1]:
        row = []
        for node in rows[-1]:
            for child in children.get(node, []):
                if child not in seen:
                    seen.add(child)
                    parent_of[child] = node
                    row.append(child)
        rows.append(row)

    position = {root: (0.0, 0.0)}
    for depth, row in enumerate(rows[1:], 1):
        for node in row:
            parent = parent_of[node]
            siblings = [n for n in row if parent_of[n] == parent]
            offset = (siblings.index(node) - (len(siblings) - 1) / 2) * 3.0
            position[node] = (position[parent][0] + offset, -2.0 * depth)
    return position


def plot_dla_tree(path: PathLike) -> Path:
    """Draw :data:`DLA_TREE_EDGES` as the ground truth the embedding should recover.

    Solid coloured edges are the branches, in the colours the embedding uses;
    faint dashed grey ones are the gaps, along which no sample exists. Matches
    the topology figure manylatents produces for the same tree, so the two can
    be compared side by side.

    Args:
        path: Where to write the PNG.

    Returns:
        ``path``, as a :class:`Path`.
    """
    from matplotlib.figure import Figure

    position = _tree_layout(DLA_TREE_EDGES)
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for from_node, to_node, edge_id, _ in DLA_TREE_EDGES:
        (x0, y0), (x1, y1) = position[from_node], position[to_node]
        if edge_id in DLA_TREE_GAPS:
            ax.plot([x0, x1], [y0, y1], color="lightgray", lw=3, alpha=0.4, ls="--")
            continue
        ax.plot([x0, x1], [y0, y1], color=_BRANCH_COLOURS[edge_id], lw=8, alpha=0.9)
        angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        angle = angle - 180 if angle > 90 else angle + 180 if angle < -90 else angle
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            str(edge_id),
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
        )

    ax.set_title("DLA tree ground truth", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    ax.margins(0.1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    return path


def genotypes_from_coordinates(
    coordinates: np.ndarray,
    n_variants: int = 1000,
    seed: int = 0,
    scale: float = 2.0,
) -> np.ndarray:
    """Genotypes whose allele frequencies drift along the coordinates.

    Each variant's log-odds of the A1 allele is a random linear function of the
    standardised coordinates, and each genotype is two draws at that frequency.
    No single variant carries the tree, but PCA over all of them recovers it,
    which is the situation the pipeline exists for.

    Args:
        coordinates: ``(n_samples, n_dim)`` float positions.
        n_variants: How many variants to sample.
        seed: Seed for the loadings and the genotype draws.
        scale: Standard deviation of the log-odds across samples. Larger means
            more of each variant's variance is structure rather than sampling.

    Returns:
        A ``(n_samples, n_variants)`` uint8 array of A1 dosages in {0, 1, 2}.
    """
    rng = np.random.default_rng(seed)
    z = (coordinates - coordinates.mean(axis=0)) / coordinates.std(axis=0)
    loadings = rng.normal(size=(z.shape[1], n_variants)) / np.sqrt(z.shape[1])
    frequency = 1.0 / (1.0 + np.exp(-scale * (z @ loadings)))
    return rng.binomial(2, frequency).astype(np.uint8)


def simulate_cohort(n_variants: int = 1000, seed: int = 0) -> Tuple[np.ndarray, pd.DataFrame]:
    """A small cohort with the population structure of a branching tree.

    Samples lie along :func:`dla_tree` -- eight connected branches, in pieces
    separated by four unsampled gaps -- and their genotypes are drawn from
    allele frequencies that drift along it. So PCA and the embedding have a
    known shape to find, with a labelled ground truth to compare against, and
    that shape is the same one manylatents uses for its own tree experiments.

    Args:
        n_variants: How many variants to genotype.
        seed: Seed for the genotypes. The tree itself is always the same one.

    Returns:
        ``(dosages, labels)`` -- an integer ``(n_samples, n_variants)`` array and
        a frame with ``sample_id`` and ``branch``.
    """
    coordinates, branch = dla_tree()
    dosages = genotypes_from_coordinates(coordinates, n_variants=n_variants, seed=seed)

    labels = pd.DataFrame(
        {
            "sample_id": [f"SIM{i:04d}" for i in range(len(branch))],
            "branch": [_BRANCH_LABEL.format(b) for b in branch],
        }
    )
    return dosages, labels


_SYNTHETIC_CONFIG = """\
# Written by `manifold-genetics acquire synthetic`.
#
# A simulated cohort of {n_samples} samples lying along a branching tree: {n_branches}
# branches, in pieces separated by {n_gaps} unsampled gaps, with genotypes drawn
# from allele frequencies that drift along it. Nothing here needs downloading,
# and the whole run takes under a minute -- it is the quickest way to confirm an
# installation works, and the embedding it makes has a known shape to check:
# the tree is drawn in {ground_truth}, beside this file.
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
  gamma: 0            # log-potential distance; branches read better on a tree than with 1

# Admixture needs the `admixture` extra (torch), so it is off by default here.
skip:
  admixture: true
"""


def acquire_synthetic(out_dir: PathLike, force: bool = False, seed: int = 0) -> Path:
    """Write a simulated cohort and a config that runs on it.

    Args:
        out_dir: Directory to write into; created if absent.
        force: Overwrite an existing ``config.yaml`` rather than refusing.
        seed: Seed for the genotypes, so the cohort is reproducible.

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
    colours = {_BRANCH_LABEL.format(b): c for b, c in _BRANCH_COLOURS.items()}
    (out_dir / "colormap.json").write_text(
        '{\n  "branch": {\n'
        + ",\n".join(f'    "{g}": "{c}"' for g, c in colours.items())
        + "\n  }\n}\n"
    )
    config_path.write_text(
        _SYNTHETIC_CONFIG.format(
            n_samples=len(labels),
            n_branches=len(_BRANCH_COLOURS),
            n_gaps=len(DLA_TREE_GAPS),
            ground_truth=GROUND_TRUTH_FIGURE,
        )
    )
    plot_dla_tree(out_dir / GROUND_TRUTH_FIGURE)

    logger.info("Wrote a simulated cohort and config to %s", out_dir)
    return config_path


def _write_keep_file(fam_path: Path, ids: pd.Series, keep_path: Path) -> None:
    """A ``plink2 --keep`` file for the samples named, as ``FID<tab>IID`` lines.

    Read from the ``.fam`` rather than written as ``id<tab>id``: ``--keep``
    matches both columns, and only the public archive has FID equal to IID. The
    workbench archive carries the population in the FID, so a keep file that
    repeated the sample ID would select nobody.
    """
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, dtype=str, usecols=[0, 1])
    keep = fam[fam[1].isin(set(ids.astype(str)))]
    keep.to_csv(keep_path, sep="\t", header=False, index=False)


def _run_plink2_keep(
    bfile: Path, keep: Path, out: Path, plink2: Optional[str], keep_chr_prefix: bool = False
) -> None:
    """``plink2 --keep --make-bed``.

    ``keep_chr_prefix`` matters for the workbench panel: plink2 writes ``chr1``
    back out as ``1`` unless told ``--output-chr chrM``, which would silently
    undo the prefix the normalisation added and that ``preprocess
    --fit-has-chr-prefix`` is then promised.
    """
    if plink2 is None:
        from .utils.tools import ToolResolver

        plink2 = ToolResolver().resolve_plink2()

    argv = [plink2, "--bfile", str(bfile), "--keep", str(keep)]
    if keep_chr_prefix:
        argv += ["--output-chr", "chrM"]
    argv += ["--make-bed", "--out", str(out), "--silent"]
    logger.info("Creating %s with plink2", out.name)
    subprocess.run(argv, check=True)


def clean(out_dir: PathLike) -> None:
    """Remove a scaffolded directory. Used by tests, not by the CLI."""
    shutil.rmtree(Path(out_dir), ignore_errors=True)


# Distinguishable at a glance and colourblind-safe enough to start from: Okabe-Ito,
# extended by cycling with varied lightness. A generated colormap is a starting
# point, not a publication choice.
_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#8C564B",
    "#7F7F7F",
)

_CUSTOM_CONFIG = """\
# Written by `manifold-genetics acquire custom`.
#
#   manifold-genetics run config.yaml --dry-run   # print the settings, do nothing
#   manifold-genetics run config.yaml             # do the work
#
# Paths are relative to this file.

# `whole_cohort` fits on the fit set and embeds the whole project set. The others
# are `projection` (fit a reference panel, project your cohort onto it) and
# `subsample` (embed a subset of a cohort too large to embed whole), which also
# supplies landmarking.
preset: {preset}

data:
  fit_plink: {fit_plink}
  project_plink: {project_plink}
{label_lines}
  colormap: colormap.json
  output_dir: outputs

pca:
  n_pcs: {n_pcs}

embedding:
  method: phate

# Admixture needs the `admixture` extra (torch), so it is off until you want it.
skip:
  admixture: true
"""


def _require_plink(prefix: Path) -> None:
    missing = [ext for ext in ("bed", "bim", "fam") if not Path(f"{prefix}.{ext}").exists()]
    if missing:
        raise FileNotFoundError(
            f"{prefix} is not a complete PLINK triple: missing "
            + ", ".join(f".{ext}" for ext in missing)
        )


def _checked_labels(labels: Path, plink: Path, min_overlap: float) -> pd.DataFrame:
    """Read a label file, refusing one that does not describe ``plink``."""
    if not labels.exists():
        raise FileNotFoundError(f"{labels} does not exist")

    frame = pd.read_csv(labels, dtype=str)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{labels} has no 'sample_id' column; found {list(frame.columns)}")

    from .pca.plink import read_fam_ids

    genotyped = set(read_fam_ids(plink))
    overlap = len(genotyped & set(frame["sample_id"])) / max(len(genotyped), 1)
    if overlap < min_overlap:
        raise ValueError(
            f"{labels} describes {overlap:.1%} of the samples in {plink}.fam. "
            f"Below {min_overlap:.0%} these are treated as different datasets: a label file "
            "that half-matches produces figures that colour half the points and look "
            "finished. Check the two describe the same cohort."
        )
    if overlap < 1.0:
        logger.warning(
            "%.1f%% of samples in %s.fam have labels; the rest will be drawn grey.",
            100 * overlap,
            plink.name,
        )
    return frame


def acquire_custom(
    out_dir: PathLike,
    fit_plink: PathLike,
    labels: Optional[PathLike] = None,
    project_plink: Optional[PathLike] = None,
    fit_labels: Optional[PathLike] = None,
    project_labels: Optional[PathLike] = None,
    preset: str = "whole_cohort",
    n_pcs: int = 20,
    force: bool = False,
    min_overlap: float = MIN_LABEL_COVERAGE,
) -> Path:
    """Write a config and colormap for genotypes you already have.

    The config is fifteen obvious lines; the colormap is not -- UK Biobank's
    needs 22 hex colours for one column -- and a value missing from it is drawn
    grey and dropped from the legend. The other trap is sample IDs: a label file
    that does not match the ``.fam`` does not crash, it colours a fraction of the
    points. Both are handled here rather than left to be discovered in a figure.

    Args:
        out_dir: Directory to write the config and colormap into.
        fit_plink: PLINK prefix the model is fitted on.
        labels: CSV with ``sample_id`` and one column per grouping, describing
            both sets. Give ``fit_labels`` and ``project_labels`` instead when
            the two cohorts are described by different files, as every UK
            Biobank config does.
        fit_labels: Labels for the fit set.
        project_labels: Labels for the project set.
        project_plink: PLINK prefix to embed. Defaults to ``fit_plink``, which is
            the common case of one cohort embedded whole.
        preset: ``whole_cohort``, ``projection`` or ``subsample``.
        n_pcs: Components to compute.
        force: Overwrite an existing ``config.yaml``.
        min_overlap: Refuse if fewer than this fraction of genotyped samples
            appear in the label file. The default is the rule every command
            applies (``preprocessing.cohort.MIN_LABEL_COVERAGE``).

    Returns:
        The path of the config file written.

    Raises:
        FileNotFoundError: a PLINK file or the label file is missing.
        ValueError: the labels do not describe this cohort.
    """
    out_dir = Path(out_dir)
    config_path = out_dir / "config.yaml"
    _refuse_to_clobber(config_path, force)

    fit_plink = Path(fit_plink)
    project_plink = Path(project_plink) if project_plink else fit_plink

    if fit_labels or project_labels:
        if not (fit_labels and project_labels):
            raise ValueError(
                "give both --fit-labels and --project-labels, or a single --labels "
                "describing both sets"
            )
        pairs = [(Path(fit_labels), fit_plink), (Path(project_labels), project_plink)]
    elif labels:
        pairs = [(Path(labels), project_plink)]
    else:
        raise ValueError("no labels given: pass --labels, or --fit-labels and --project-labels")

    # Everything is checked before anything is written, so a rejected run leaves
    # no half-made directory to puzzle over.
    _require_plink(fit_plink)
    _require_plink(project_plink)

    frames = []
    for label_path, plink in pairs:
        frames.append(_checked_labels(label_path, plink, min_overlap))
    label_frame = pd.concat(frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_generated_colormap(label_frame, out_dir / "colormap.json")

    if fit_labels and project_labels:
        label_lines = f"  fit_labels: {fit_labels}\n  project_labels: {project_labels}"
    else:
        label_lines = f"  labels: {labels}"

    config_path.write_text(
        _CUSTOM_CONFIG.format(
            preset=preset,
            fit_plink=fit_plink,
            project_plink=project_plink,
            label_lines=label_lines,
            n_pcs=n_pcs,
        )
    )
    logger.info("Wrote a config for %s to %s", fit_plink.name, out_dir)
    return config_path


def _write_generated_colormap(labels: pd.DataFrame, path: Path) -> None:
    """A colour for every value of every label column, so no point goes grey.

    Generated, therefore provisional: it is the file you recolour for a figure,
    and it exists so that nobody hand-writes 22 hex codes to find out whether
    their pipeline runs.
    """
    colormap = {}
    for column in labels.columns:
        if column == "sample_id":
            continue
        values = sorted(labels[column].dropna().astype(str).unique())
        colormap[column] = {value: _PALETTE[i % len(_PALETTE)] for i, value in enumerate(values)}

    if not colormap:
        raise ValueError("the label file has no columns besides sample_id to colour by")

    import json as _json

    path.write_text(_json.dumps(colormap, indent=2) + "\n")
