"""``manifold-genetics acquire hgdp``: fetch and prepare the HGDP+1KGP cohort.

Ported from ``examples/hgdp_1kgp/download_data.sh`` and ``prepare_data.sh``, so
what ``acquire hgdp`` writes is the cohort the published figures were made from.
It also accepts the archive the All of Us workbench keeps beside its data,
which is a different, unfiltered panel, and normalises it into the same layout.

Lives in its own module for the same reason ``aou`` does: it is long enough to
deserve one. ``scaffold`` re-exports every name here, so ``from
manifold_genetics.scaffold import acquire_hgdp`` keeps working; the generic
plink helpers it calls (``_write_keep_file``, ``_run_plink2_keep``) stay in
``scaffold`` because ``subsample`` uses them too.
"""

import json
import logging
import tarfile
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from .aou import _run_gsutil
from .utils.tools import fetch_url

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# The same archive examples/hgdp_1kgp/download_data.sh fetches: full_dataset
# .bed/.bim/.fam (4,151 samples, 172,152 SNPs) plus metadata.csv.
HGDP_ARCHIVE_URL = (
    "https://www.dropbox.com/scl/fi/f8kay7kgpgb2s5ncsykpg/hgdp_1kgp_full.tar.gz"
    "?rlkey=pp7fj711zxlg4o2lbskxj5rwe&st=jv1xh8vo&dl=1"
)


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
        fetch_url(HGDP_ARCHIVE_URL, archive)
        return archive
    except OSError as exc:
        # Most often a TLS-intercepting proxy, whose root certificate Python
        # does not trust, or a host with no route out at all -- an HPC compute
        # node, say. Neither is fixable from here, but both are worked around
        # the same way, so the error has to say how rather than suggest a retry.
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the HGDP+1KGP archive: {exc}\n\n"
            "If this is a proxy or an offline machine, fetch it by other means and "
            "point acquire at the file:\n\n"
            f"    curl -L -o hgdp_1kgp_full.tar.gz '{HGDP_ARCHIVE_URL}'\n"
            "    manifold-genetics acquire hgdp --archive hgdp_1kgp_full.tar.gz\n\n"
            "curl and wget use the system certificate store, which on a managed "
            "network usually already trusts the proxy."
        ) from exc
    return archive


# The two HGDP+1KGP archives `acquire hgdp` knows how to unpack. They are not
# the same panel: the public one is filtered and LD-pruned and carries QC and
# relatedness flags in metadata.csv; the workbench one is unfiltered and carries
# nothing but the population, in the FID.
_PUBLIC_PREFIX = "full_dataset"
_WORKBENCH_PREFIX = "extractedChrAllUnpruned"
_WORKBENCH_FID_PREFIX = "forReference"


def _detect_hgdp_layout(raw_dir: Path) -> str:
    """Which HGDP+1KGP archive was unpacked here.

    "public" is the Dropbox archive `acquire hgdp` fetches: ``full_dataset.*``,
    already filtered and LD-pruned, with a ``metadata.csv``. "workbench" is the
    archive kept beside All of Us: ``extractedChrAllUnpruned.*``, unfiltered,
    chromosomes without a ``chr`` prefix, and the population carried in the FID
    as ``forReference<Population>``. They are not the same panel and the log
    line says which one this is.

    The workbench check comes first, and accepts the ``.bim`` as well as the
    ``.bed``: normalising a workbench archive renames its ``.bed`` to the public
    name but leaves the original ``.bim`` and ``.fam`` in place, so a directory
    that has been normalised -- or is being re-run -- still says where its data
    came from. Without that, every re-run would look public and go looking for
    QC columns the workbench metadata does not have.
    """
    if any((raw_dir / f"{_WORKBENCH_PREFIX}{ext}").exists() for ext in (".bed", ".bim")):
        return "workbench"
    if (raw_dir / f"{_PUBLIC_PREFIX}.bed").exists():
        return "public"
    raise FileNotFoundError(
        f"{raw_dir} holds neither {_PUBLIC_PREFIX}.bed (the public archive) nor "
        f"{_WORKBENCH_PREFIX}.bed (the workbench archive)."
    )


def _normalise_workbench_layout(raw_dir: Path) -> None:
    """Rewrite the workbench archive into the public layout.

    What ``examples/aou/hgdp_1kgp_proj/prepare_data.sh`` did in steps 3 and 14:
    ``chr`` on every chromosome, and the population out of the FID and into a
    ``metadata.csv``. The ``.bed`` is renamed, never read -- it is the same
    genotypes under the other name.
    """
    src = raw_dir / _WORKBENCH_PREFIX
    dst = raw_dir / _PUBLIC_PREFIX

    bim = pd.read_csv(f"{src}.bim", sep=r"\s+", header=None, dtype=str)
    bim[0] = bim[0].where(bim[0].str.startswith("chr"), "chr" + bim[0])
    bim.to_csv(f"{dst}.bim", sep="\t", header=False, index=False)

    fam = pd.read_csv(f"{src}.fam", sep=r"\s+", header=None, dtype=str)
    population = fam[0].str.replace(f"^{_WORKBENCH_FID_PREFIX}", "", regex=True)
    fam[0] = population
    fam.to_csv(f"{dst}.fam", sep="\t", header=False, index=False)

    Path(f"{src}.bed").rename(f"{dst}.bed")
    pd.DataFrame({"project_meta.sample_id": fam[1], "Population": population}).to_csv(
        raw_dir / "metadata.csv", index=False
    )
    logger.warning(
        "Workbench HGDP+1KGP archive: unfiltered, %d samples. Not the public panel; "
        "run `preprocess --preset harmonise --fit-has-chr-prefix` before fitting on it.",
        len(fam),
    )


def _extract_hgdp_archive(archive: Path, raw_dir: Path) -> None:
    """Unpack the archive into ``raw_dir`` in the public layout.

    The tar is flattened by basename -- the workbench archive nests its files
    under a ``1KGPHGDP/`` directory, the public one does not -- and a workbench
    archive is then normalised, so what follows sees ``full_dataset.*`` and a
    ``metadata.csv`` either way. Already-extracted data is left alone.
    """
    if (raw_dir / f"{_PUBLIC_PREFIX}.bed").exists():
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

    layout = _detect_hgdp_layout(raw_dir)
    logger.info("Unpacked the %s HGDP+1KGP archive", layout)
    if layout == "workbench":
        _normalise_workbench_layout(raw_dir)


# The comment block at the top of the config says what the data beside it is,
# which differs by archive; the settings below it do not.
_HGDP_PUBLIC_HEADER = """\
# Written by `manifold-genetics acquire hgdp`.
#
# HGDP+1KGP: 4,094 QC-passing samples across seven genetic regions, with the
# model fitted on the 3,400 that are also unrelated. This is the cohort the
# published figures were made from.
#
"""
_HGDP_WORKBENCH_HEADER = """\
# Written by `manifold-genetics acquire hgdp`, from the workbench archive.
#
# HGDP+1KGP as kept beside All of Us: every sample, unfiltered, labelled by
# population. Not the public panel the published figures were made from. Run
# `preprocess --preset harmonise --fit-has-chr-prefix` before fitting on it.
#
"""
# ``{geographic_line}`` is either empty or a ``geographic_coords:`` line under
# ``data:`` -- present when a geographic.csv was written, which the workbench
# archive's metadata (no coordinates) never gets. See _write_hgdp_geographic.
_HGDP_CONFIG_BODY = """\
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
{geographic_line}  output_dir: outputs

pca:
  n_pcs: 20

admixture:
  k_min: 2
  k_max: 10

embedding:
  method: phate

visualization:
  admix_group_column: Genetic_region_merged

# The settings above are those of the published run. Admixture needs the
# `admixture` extra (torch) and is slow without a GPU: delete this `skip` line
# (or set it to false) once `pip install 'manifold-genetics[admixture]'` is
# done and you are on a GPU node.
skip:
  admixture: true
"""
_HGDP_CONFIG = _HGDP_PUBLIC_HEADER + _HGDP_CONFIG_BODY.format(geographic_line="")


def _hgdp_config(layout: str, geographic: bool = False) -> str:
    header = _HGDP_PUBLIC_HEADER if layout == "public" else _HGDP_WORKBENCH_HEADER
    geographic_line = "  geographic_coords: data/geographic.csv\n" if geographic else ""
    return header + _HGDP_CONFIG_BODY.format(geographic_line=geographic_line)


def acquire_hgdp(
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
            of fetching, for proxied networks and offline machines -- or a
            ``gs://`` URL, fetched with gsutil into ``data/`` once. Either the
            public archive or the workbench's ``1KGPHGDP.tar.gz``; the layout
            is detected and the workbench one normalised.

    Returns:
        The path of the config file written.
    """
    # scaffold re-exports this module's names, so its helpers are imported here
    # rather than at the top -- and looked up through the module at call time,
    # as subsample._plink2_keep does, so a test that patches
    # scaffold._run_plink2_keep is honoured.
    from . import scaffold

    out_dir = Path(out_dir)
    config_path = out_dir / "config.yaml"
    scaffold._refuse_to_clobber(config_path, force)

    data_dir = out_dir / "data"
    raw_dir = data_dir / "raw"

    if not (raw_dir / "full_dataset.bed").exists():
        # Prefer a file we already have over the network, in this order: one the
        # caller named, one left in place by an earlier run, then downloading.
        # `download=False` means "do not fetch", not "do not unpack" -- someone
        # who obtained the archive another way still needs it extracted.
        if archive and str(archive).startswith("gs://"):
            # The workbench keeps its copy in a bucket; gsutil is the only way in.
            fetched = data_dir / Path(str(archive)).name
            if not fetched.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Fetching %s with gsutil", archive)
                _run_gsutil(["gsutil", "cp", str(archive), str(fetched)])
            archive = fetched
        if archive:
            # A named archive is the one to use. Falling back to the public
            # download when it is missing would silently swap in a different
            # panel, which is worse than stopping.
            local = Path(archive)
            if not local.exists():
                raise FileNotFoundError(f"The archive named, {local}, does not exist.")
            _extract_hgdp_archive(local, raw_dir)
        else:
            local = data_dir / "hgdp_1kgp_full.tar.gz"
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

    layout = _detect_hgdp_layout(raw_dir)
    metadata = pd.read_csv(raw_dir / "metadata.csv")
    if layout == "public":
        fit_ids, project_ids = hgdp_subsets(metadata)
    else:
        # The workbench panel carries no relatedness or QC columns, so there is
        # no unrelated subset to pick; the old AoU flow fitted on every sample,
        # and so does this.
        fit_ids = project_ids = metadata["project_meta.sample_id"]
    logger.info("Selected %d fit and %d project samples", len(fit_ids), len(project_ids))

    for name, ids in (("fit", fit_ids), ("project", project_ids)):
        keep = data_dir / f"{name}_indices.txt"
        scaffold._write_keep_file(raw_dir / "full_dataset.fam", ids, keep)
        scaffold._run_plink2_keep(
            raw_dir / "full_dataset",
            keep,
            data_dir / f"{name}_subset",
            plink2,
            keep_chr_prefix=layout == "workbench",
        )

    if layout == "public":
        _write_hgdp_labels(
            metadata, project_ids, data_dir / "labels.csv", out_dir / "colormap.json"
        )
    else:
        labels = metadata.rename(columns={"project_meta.sample_id": "sample_id"})
        labels.to_csv(data_dir / "labels.csv", index=False)
        scaffold._write_generated_colormap(labels, out_dir / "colormap.json")
    wrote_geographic = _write_hgdp_geographic(metadata, project_ids, data_dir / "geographic.csv")
    config_path.write_text(_hgdp_config(layout, geographic=wrote_geographic))

    logger.info("Wrote HGDP+1KGP and a config to %s", out_dir)
    return config_path


# The metadata columns naming each sample's population and genetic region, and
# the colours the shipped example uses for them -- examples/colormaps/hgdp_1kgp.json
# -- so a figure from `acquire hgdp` is comparable with the published ones.
_HGDP_POPULATION_COLUMN = "Population"
_HGDP_REGION_COLUMN = "Genetic_region_merged"
_HGDP_REGION_COLOURS = {
    "Africa": "#008000",
    "America": "#FF0000",
    "Central_South_Asia": "#FFA500",
    "East_Asia": "#0000FF",
    "Europe": "#800080",
    "Middle_East": "#808080",
    "Oceania": "#FFFF00",
}
# The 78 population colours from the published run's colormap, copied verbatim
# from examples/colormaps/hgdp_1kgp.json (see test_the_colormap_has_a_published_
# colour_for_every_population, which checks this against that file).
_HGDP_POPULATION_COLOURS = {
    "ACB": "#006D2C",
    "ASW": "#00441B",
    "BantuKenya": "#008000",
    "BantuSouthAfrica": "#008000",
    "BiakaPygmy": "#008000",
    "ESN": "#A1D99B",
    "GWD": "#74C476",
    "LWK": "#41AB5D",
    "MSL": "#238B45",
    "Mandenka": "#008000",
    "MbutiPygmy": "#008000",
    "San": "#008000",
    "YRI": "#C7E9C0",
    "Yoruba": "#008000",
    "CLM": "#E3242B",
    "Colombian": "#FF0000",
    "Karitiana": "#FF0000",
    "MXL": "#BC544B",
    "Maya": "#FF0000",
    "PEL": "#E0115F",
    "PUR": "#900D09",
    "Pima": "#FF0000",
    "Surui": "#FF0000",
    "BEB": "#FDBE85",
    "Balochi": "#FFA500",
    "Brahui": "#FFA500",
    "Burusho": "#FFA500",
    "GIH": "#FD8D3C",
    "Hazara": "#FFA500",
    "ITU": "#E6550D",
    "Kalash": "#FFA500",
    "Makrani": "#FFA500",
    "PJL": "#FEEDDE",
    "Pathan": "#FFA500",
    "STU": "#E6550D",
    "Sindhi": "#FFA500",
    "CDX": "#008080",
    "CHB": "#DEEBF7",
    "CHS": "#9ECAE1",
    "Cambodian": "#0000FF",
    "Dai": "#0000FF",
    "Daur": "#0000FF",
    "Han": "#0000FF",
    "Hezhen": "#0000FF",
    "JPT": "#08519C",
    "Japanese": "#0000FF",
    "KHV": "#0ABAB5",
    "Lahu": "#0000FF",
    "Miao": "#0000FF",
    "Mongola": "#0000FF",
    "Naxi": "#0000FF",
    "Oroqen": "#0000FF",
    "She": "#0000FF",
    "Tu": "#0000FF",
    "Tujia": "#0000FF",
    "Uygur": "#0000FF",
    "Xibo": "#0000FF",
    "Yakut": "#0000FF",
    "Yi": "#0000FF",
    "Adygei": "#800080",
    "Basque": "#800080",
    "CEU": "#D896FF",
    "FIN": "#800080",
    "French": "#800080",
    "GBR": "#D896FF",
    "IBS": "#EFBBFF",
    "Italian": "#800080",
    "Orcadian": "#800080",
    "Russian": "#800080",
    "Sardinian": "#800080",
    "TSI": "#BE29EC",
    "Tuscan": "#800080",
    "Bedouin": "#808080",
    "Druze": "#808080",
    "Mozabite": "#808080",
    "Palestinian": "#808080",
    "Melanesian": "#FFFF00",
    "Papuan": "#FFFF00",
}


def _write_hgdp_labels(
    metadata: pd.DataFrame, project_ids: pd.Series, labels_path: Path, colormap_path: Path
) -> None:
    """Write the label file and colormap for the prepared cohort.

    Raises rather than substituting a placeholder when the population or region
    column is absent. It previously wrote "Unknown" for every sample, which
    produced a complete run, a drawn figure, and every point in it the same grey
    -- a failure that looks like success until someone reads the legend.
    """
    for column in (_HGDP_POPULATION_COLUMN, _HGDP_REGION_COLUMN):
        if column not in metadata.columns:
            raise KeyError(
                f"{column!r} is not a column of the cohort metadata "
                f"(found: {sorted(metadata.columns)[:8]}...). Without it the samples "
                "cannot be labelled, and an unlabelled figure is worse than none."
            )

    keep = metadata[metadata["project_meta.sample_id"].isin(set(project_ids))]
    population = keep[_HGDP_POPULATION_COLUMN].astype(str)
    region = keep[_HGDP_REGION_COLUMN].astype(str)

    pd.DataFrame(
        {
            "sample_id": keep["project_meta.sample_id"],
            _HGDP_POPULATION_COLUMN: population,
            _HGDP_REGION_COLUMN: region,
        }
    ).to_csv(labels_path, index=False)

    colormap = {}
    for column, values, published in (
        (_HGDP_POPULATION_COLUMN, population, _HGDP_POPULATION_COLOURS),
        (_HGDP_REGION_COLUMN, region, _HGDP_REGION_COLOURS),
    ):
        unknown = sorted(set(values) - set(published))
        if unknown:
            logger.warning("No published colour for %s; drawn in grey", ", ".join(unknown))
        colormap[column] = {v: published.get(v, "#999999") for v in sorted(set(values))}

    colormap_path.write_text(json.dumps(colormap, indent=2) + "\n")


# Populations that do not preserve geography well -- African-American and
# Afro-Caribbean populations, and Utah residents with European ancestry, all
# admixed enough that a geographic embedding metric would be meaningless for
# them -- excluded the same way examples/hgdp_1kgp/prepare_data.sh step 9 did.
_HGDP_GEOGRAPHY_EXCLUDED_POPULATIONS = frozenset({"ACB", "ASW", "CEU"})


def _write_hgdp_geographic(metadata: pd.DataFrame, project_ids: pd.Series, path: Path) -> bool:
    """Write ``sample_id, latitude, longitude`` for the project set, less the
    samples whose geography is not expected to be preserved.

    Excludes ``Genetic_region_merged == "America"`` and
    :data:`_HGDP_GEOGRAPHY_EXCLUDED_POPULATIONS`, exactly as
    ``examples/hgdp_1kgp/prepare_data.sh`` step 9 did. The workbench archive's
    metadata carries no coordinates at all, so this writes nothing for it --
    returning False rather than raising, since geographic metrics are optional
    and the rest of the run does not depend on them.

    Returns:
        Whether the file was written.
    """
    if "latitude" not in metadata.columns or "longitude" not in metadata.columns:
        logger.warning(
            "No latitude/longitude columns in the cohort metadata; skipping "
            "geographic coordinates (geographic_coords will not appear in the config)."
        )
        return False

    keep = metadata[metadata["project_meta.sample_id"].isin(set(project_ids))]
    excluded = pd.Series(False, index=keep.index)
    if _HGDP_REGION_COLUMN in keep.columns:
        excluded |= keep[_HGDP_REGION_COLUMN] == "America"
    if _HGDP_POPULATION_COLUMN in keep.columns:
        excluded |= keep[_HGDP_POPULATION_COLUMN].isin(_HGDP_GEOGRAPHY_EXCLUDED_POPULATIONS)

    geo = (
        keep.loc[~excluded, ["project_meta.sample_id", "latitude", "longitude"]]
        .rename(columns={"project_meta.sample_id": "sample_id"})
        .dropna(subset=["latitude", "longitude"])
    )
    geo.to_csv(path, index=False)
    return True
