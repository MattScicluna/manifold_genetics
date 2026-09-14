"""All of Us: fetch the cohort inside the Researcher Workbench.

A port of ``examples/aou/shared/download_aou_data.sh`` -- the GCS download, the
FAM fix and the BigQuery metadata extraction -- plus the project-labels block
of ``examples/aou/hgdp_1kgp_proj/prepare_data.sh`` (step 14), which turned that
metadata into a label file. Each step is a function taking its inputs and the
``runner``/``query`` it calls out through, so the sequence is testable without
GCS or BigQuery; each is skipped when its output exists, as the script's own
``if [[ ! -f ... ]]`` checks did.

What comes out is a *whole_cohort* cohort directory: the cohort alone. The
published analysis projected it onto HGDP+1KGP; that is now two more commands
(``acquire hgdp --archive gs://...`` and ``preprocess --preset harmonise``),
which is what the config's comment says.

The script's per-chromosome split (its step 5, ``plink --chr N`` for 1..22) is
deliberately not ported: it was "optional, for parallel processing", and the
downstream ``prepare_data.sh`` reads ``extractedChrAll`` directly. Nothing in
the pipeline consumes ``extractedChr{1..22}``, so ``acquire aou`` needs no
plink at all.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Union

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# What the All of Us preparation needs, and where each comes from. Checked
# together rather than one at a time: reporting them singly would be three round
# trips for someone who is simply not in the workbench.
_AOU_REQUIRED_ENV = {
    "GOOGLE_PROJECT": "the workbench's billing project, set for you inside it",
    "WORKSPACE_CDR": "the CDR version, used to read demographics from BigQuery",
}
_AOU_REQUIRED_TOOLS = {
    "gsutil": "to read gs://fc-aou-datasets-controlled",
    "bq": "to query the CDR for sample demographics",
}

# download_aou_data.sh: AOU_BUCKET_ROOT_V8. The bucket calls the files
# arrays.*; the script renamed them to extractedChrAll.* on the way down, and
# so does this, so a directory made either way reads the same.
AOU_BUCKET_ROOT_V8 = "gs://fc-aou-datasets-controlled/v8/microarray/plink"
_BUCKET_PREFIX = "arrays"
_LOCAL_PREFIX = "extractedChrAll"

_AOU_CONFIG = """\
# Written by `manifold-genetics acquire aou`.
#
# All of Us V8 microarray genotypes, every sample, labelled by self-reported
# race and ethnicity from the CDR. This is the cohort alone. The published
# analysis projected it onto the HGDP+1KGP panel kept beside it in the
# workbench; to reproduce that, acquire the panel and intersect the two:
#
#   manifold-genetics acquire hgdp --archive gs://.../1KGPHGDP.tar.gz --out ref/
#   manifold-genetics preprocess ref/config.yaml config.yaml \\
#       --preset harmonise --fit-has-chr-prefix --out proj/
#
#   manifold-genetics run config.yaml --dry-run   # print the settings, do nothing
#   manifold-genetics run config.yaml             # do the work
#
# Paths are relative to this file. data/project_subset.* are links to the
# download under data/raw/, whose .fam has been given a family ID (AOU) and a
# missing phenotype (-9) as the array files in the bucket carry neither.

preset: whole_cohort

data:
  fit_plink: data/project_subset
  project_plink: data/project_subset
  labels: data/labels.csv
  colormap: colormap.json
  output_dir: outputs

pca:
  n_pcs: 20

embedding:
  method: phate
  # 400,000-odd samples are transformed; batching keeps peak memory bounded.
  embed_batch_size: 60000

# Admixture needs the `admixture` extra (torch) and a GPU to be worth running.
skip:
  admixture: true
"""


def aou_environment_problems() -> list:
    """What is missing before All of Us data can be prepared, in one pass."""
    problems = []
    for variable, why in _AOU_REQUIRED_ENV.items():
        if not os.environ.get(variable):
            problems.append(f"  ${variable} is not set -- {why}")
    for tool, why in _AOU_REQUIRED_TOOLS.items():
        if not shutil.which(tool):
            problems.append(f"  {tool} is not on PATH -- {why}")
    return problems


def _read_gbq(sql: str) -> pd.DataFrame:
    """The script's ``pd.read_gbq(sql, dialect="standard")``, imported lazily:
    pandas-gbq is the ``aou`` extra, and nothing else in the package needs it."""
    try:
        import pandas_gbq
    except ImportError as exc:
        raise ImportError(
            "Reading All of Us demographics from BigQuery needs pandas-gbq, which is "
            "the `aou` extra: pip install 'manifold-genetics[aou]'"
        ) from exc
    return pandas_gbq.read_gbq(sql, dialect="standard")


# =============================================================================
# STEP 1: All of Us Genotype Download (V8)
# =============================================================================
def _fetch_plink(bucket_root: str, project: str, raw_dir: Path, runner: Callable) -> Path:
    """Download the PLINK files, each unless already present.

    The script: for each extension, ``gsutil -u "$GOOGLE_PROJECT" cp
    "$SRC" "$DEST"`` when ``$DEST`` is missing. ``-u`` bills the requester-pays
    bucket to the workbench's project.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("bed", "bim", "fam"):
        src = f"{bucket_root}/{_BUCKET_PREFIX}.{ext}"
        dest = raw_dir / f"{_LOCAL_PREFIX}.{ext}"
        if dest.exists():
            logger.info("  %s file already exists", ext)
            continue
        logger.info("Downloading %s file...", ext)
        runner(["gsutil", "-u", project, "cp", src, str(dest)], check=True)
    return raw_dir / _LOCAL_PREFIX


def _fix_fam(prefix: Path) -> Path:
    """Fix the FAM file: add a family ID, and mark the phenotype missing.

    The script: ``mv`` the download to ``.Original.fam``, then
    ``awk '{print "AOU\\t"$2"\\t"$3"\\t"$4"\\t"$5"\\t-9"}'`` over it. The
    ``.Original.fam`` is the marker that it has been done, so a second run
    leaves both files alone.
    """
    fam_path = prefix.with_suffix(".fam")
    orig_path = prefix.parent / f"{prefix.name}.Original.fam"
    if orig_path.exists():
        logger.info("  FAM file already fixed")
        return fam_path

    logger.info("Fixing AoU FAM file (adding Family ID)...")
    fam_path.rename(orig_path)
    with orig_path.open() as src, fam_path.open("w") as dest:
        for line in src:
            fields = line.split()
            if not fields:
                continue
            dest.write("\t".join(["AOU", *fields[1:5], "-9"]) + "\n")
    return fam_path


# =============================================================================
# STEP 2: Metadata Extraction
# =============================================================================
def _fetch_metadata(cdr: str, meta_dir: Path, query: Callable) -> Path:
    """The script's embedded ``extract_metadata.py``, run when either file is
    missing (the script skipped it only when both were present).

    Both SQL strings are the script's, verbatim, with the CDR version
    substituted from ``$WORKSPACE_CDR``.
    """
    demographics = meta_dir / "DemographicData.tsv"
    ses = meta_dir / "SocioeconomicZipCodes.tsv"
    if demographics.exists() and ses.exists():
        logger.info("Metadata files already exist, skipping extraction")
        return demographics
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Query 1: Demographics
    logger.info("Querying Demographics...")
    sql_person = f"""
    SELECT
        person.person_id,
        person.gender_concept_id,
        p_gender_concept.concept_name as gender,
        person.birth_datetime as date_of_birth,
        person.race_concept_id,
        p_race_concept.concept_name as race,
        person.ethnicity_concept_id,
        p_ethnicity_concept.concept_name as ethnicity,
        person.sex_at_birth_concept_id,
        p_sex_at_birth_concept.concept_name as sex_at_birth
    FROM `{cdr}.person` person
    LEFT JOIN `{cdr}.concept` p_gender_concept
        ON person.gender_concept_id = p_gender_concept.concept_id
    LEFT JOIN `{cdr}.concept` p_race_concept
        ON person.race_concept_id = p_race_concept.concept_id
    LEFT JOIN `{cdr}.concept` p_ethnicity_concept
        ON person.ethnicity_concept_id = p_ethnicity_concept.concept_id
    LEFT JOIN `{cdr}.concept` p_sex_at_birth_concept
        ON person.sex_at_birth_concept_id = p_sex_at_birth_concept.concept_id
"""
    df_person = query(sql_person)
    df_person = _derive_race_ethnicity(df_person)
    df_person.to_csv(demographics, sep="\t", index=False)
    logger.info("  Saved %d demographic records", len(df_person))

    # Query 2: Socioeconomics
    logger.info("Querying Socioeconomics...")
    sql_ses = f"""
    SELECT
        observation.person_id,
        zip_code.zip3_as_string as zip_code,
        zip_code.fraction_assisted_income as assisted_income,
        zip_code.fraction_high_school_edu as high_school_education,
        zip_code.median_income,
        zip_code.fraction_no_health_ins as no_health_insurance,
        zip_code.fraction_poverty as poverty,
        zip_code.deprivation_index
    FROM `{cdr}.zip3_ses_map` zip_code
    JOIN `{cdr}.observation` observation
        ON CAST(SUBSTR(observation.value_as_string, 0, STRPOS(observation.value_as_string, '*') - 1) AS INT64) = zip_code.zip3
        AND observation_source_concept_id = 1585250
        AND observation.value_as_string NOT LIKE 'Res%'
"""  # noqa: E501
    df_ses = query(sql_ses)
    df_ses.to_csv(ses, sep="\t", index=False)
    logger.info("  Saved %d socioeconomic records", len(df_ses))
    return demographics


def _derive_race_ethnicity(df_person: pd.DataFrame) -> pd.DataFrame:
    """Preprocess race and ethnicity columns, as ``extract_metadata.py`` did.

    Uninformative answers in either column become "No information".
    ``race_ethnicity`` is the race, except that "Hispanic or Latino" is
    assigned when the race is "No information" and the ethnicity is
    "Hispanic or Latino".
    """
    df_person = df_person.copy()
    race_no_info = ["I prefer not to answer", "None of these", "PMI: Skip", "None Indicated"]
    df_person["race"] = df_person["race"].replace(race_no_info, "No information")

    ethnicity_no_info = [
        "PMI: Prefer Not To Answer",
        "What Race Ethnicity: Race Ethnicity None Of These",
        "PMI: Skip",
    ]
    df_person["ethnicity"] = df_person["ethnicity"].replace(ethnicity_no_info, "No information")

    df_person["race_ethnicity"] = df_person["race"]
    hispanic_mask = (df_person["race"] == "No information") & (
        df_person["ethnicity"] == "Hispanic or Latino"
    )
    df_person.loc[hispanic_mask, "race_ethnicity"] = "Hispanic or Latino"
    return df_person


# =============================================================================
# prepare_data.sh step 14: project labels
# =============================================================================
_LABEL_COLUMNS = ("race", "ethnicity", "race_ethnicity")


def _write_labels(demographics: Path, fam_path: Path, labels_path: Path) -> pd.DataFrame:
    """The metadata rows that are in the .fam, with ``person_id`` as ``sample_id``
    and the race columns that are present -- ``prepare_data.sh``'s
    ``columns_to_keep``.

    Written every run: reusing the file because it exists is how
    ukbb/geosketch_phate's labels went stale, and the inputs are on disk.
    """
    metadata = pd.read_csv(demographics, sep="\t", low_memory=False)
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, usecols=[1], names=["IID"], dtype=str)
    available = set(fam["IID"])

    metadata["sample_id"] = metadata["person_id"].astype(str)
    labels = metadata[metadata["sample_id"].isin(available)].copy()
    columns = ["sample_id"] + [c for c in _LABEL_COLUMNS if c in labels.columns]
    labels = labels[columns]
    labels.to_csv(labels_path, index=False)
    logger.info("  Saved %s: %d samples", labels_path.name, len(labels))

    if len(labels) < len(available):
        logger.warning(
            "%d samples in the genotypes have no metadata and will be unlabelled",
            len(available) - len(labels),
        )
    return labels


def _link_project_subset(prefix: Path, data_dir: Path) -> Path:
    """``data/project_subset.*`` pointing at the download: the .bed is large,
    and a link with an absolute target reads correctly from anywhere."""
    subset = data_dir / "project_subset"
    for ext in ("bed", "bim", "fam"):
        link = subset.with_suffix(f".{ext}")
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(prefix.with_suffix(f".{ext}").resolve())
    return subset


def acquire_aou(
    out_dir: PathLike,
    force: bool = False,
    *,
    runner: Callable = subprocess.run,
    query: Callable = _read_gbq,
) -> Path:
    """Fetch All of Us from the workbench bucket and write a config that runs on it.

    The steps of ``download_aou_data.sh`` in order -- download, FAM fix,
    metadata -- then the labels ``prepare_data.sh`` made from that metadata.
    Every step reuses what an earlier run left behind, so a run that was
    interrupted picks up where it stopped.

    Args:
        out_dir: Directory to write into.
        force: Overwrite an existing ``config.yaml``.
        runner: What runs ``gsutil``; ``subprocess.run`` outside tests.
        query: What runs a BigQuery SQL string and returns a DataFrame;
            ``pandas_gbq.read_gbq`` (the ``aou`` extra) outside tests.

    Returns:
        The path of the config file written.

    Raises:
        EnvironmentError: this is not an All of Us Researcher Workbench.
        subprocess.CalledProcessError: a ``gsutil`` copy failed.
    """
    # scaffold re-exports this module's names, so its helpers are imported here
    # rather than at the top.
    from .scaffold import _refuse_to_clobber, _write_generated_colormap

    out_dir = Path(out_dir)
    config_path = out_dir / "config.yaml"
    _refuse_to_clobber(config_path, force)

    problems = aou_environment_problems()
    if problems:
        raise EnvironmentError(
            "All of Us data can only be prepared inside the Researcher Workbench, "
            "and this does not look like one:\n\n"
            + "\n".join(problems)
            + "\n\nIf you are in the workbench, these are normally set for you; "
            "check the notebook environment. If you are not, there is no way to "
            "reach the data from here -- it is controlled-access.\n\n"
            "To try the pipeline without it: `manifold-genetics acquire synthetic`, "
            "or `acquire hgdp` for a real public cohort."
        )

    data_dir = out_dir / "data"
    raw_dir = data_dir / "raw"
    meta_dir = raw_dir / "Metadata"

    prefix = _fetch_plink(AOU_BUCKET_ROOT_V8, os.environ["GOOGLE_PROJECT"], raw_dir, runner)
    fam_path = _fix_fam(prefix)
    demographics = _fetch_metadata(os.environ["WORKSPACE_CDR"], meta_dir, query)

    subset = _link_project_subset(prefix, data_dir)
    labels = _write_labels(demographics, fam_path, data_dir / "labels.csv")
    _write_generated_colormap(labels, out_dir / "colormap.json")
    config_path.write_text(_AOU_CONFIG)

    logger.info("Wrote All of Us (%s) and a config to %s", subset.name, out_dir)
    return config_path


__all__ = ["acquire_aou", "aou_environment_problems", "AOU_BUCKET_ROOT_V8"]
