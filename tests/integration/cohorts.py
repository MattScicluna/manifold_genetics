"""Which real cohorts the pipeline is tested against, and where their data is.

Four of the seven shipped examples need controlled-access genotypes that no
clean checkout has, and two of those (All of Us) can only be run inside a secure
workbench. So the tests that use them have to skip cleanly rather than fail, and
nothing here may hardcode a cluster path.

Each cohort names an environment variable holding its data root. Unset, the root
falls back to the example's own directory, which is what makes the whole suite
work with no configuration on a machine where the examples are already prepared.

The registry is deliberately a *statement of intent*: every shipped example must
appear here, so adding an example that is never exercised against real data
fails a test instead of passing unnoticed.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

REPO = Path(__file__).resolve().parents[2]

# Every example needs its two PLINK triples. Listing all six extensions rather
# than just the .bed catches a half-copied dataset, which otherwise surfaces as
# an opaque failure deep inside flashpca.
_PLINK = tuple(
    f"data/{stem}_subset.{ext}" for stem in ("fit", "project") for ext in ("bed", "bim", "fam")
)


@dataclass(frozen=True)
class Cohort:
    """One real dataset the pipeline is tested against.

    ``config`` is repo-relative so it can be compared against the shipped
    examples; ``requires`` is relative to the resolved data root, because that
    root moves when the environment variable is set.
    """

    name: str
    config: str
    env_var: str
    public: bool
    requires: Tuple[str, ...]

    @property
    def config_path(self) -> Path:
        return REPO / self.config

    def missing(self, root: Path) -> list:
        """The required files absent under ``root``, in declaration order."""
        return [rel for rel in self.requires if not (root / rel).exists()]


def resolve_data_root(cohort: Cohort) -> Optional[Path]:
    """Where ``cohort``'s data lives, or None if that location does not exist.

    A set-but-wrong environment variable returns None rather than silently
    falling back: someone who sets it meant to point the test somewhere, and
    quietly testing a different dataset instead is worse than skipping.
    """
    override = os.environ.get(cohort.env_var)
    if override:
        root = Path(override)
        return root if root.is_dir() else None

    root = cohort.config_path.parent
    return root if root.is_dir() else None


def rebase(path, cohort: Cohort, root: Path) -> Path:
    """Move ``path`` from the example directory onto ``root``.

    A config's paths resolve against the config file, but the data root can be
    somewhere else entirely -- an All of Us workbench bucket, a scratch copy.
    Paths that do not live under the example (colormaps, which are tracked in
    the repo) are returned unchanged, because they are not part of anybody's
    private data.
    """
    path = Path(path)
    try:
        relative = path.relative_to(cohort.config_path.parent)
    except ValueError:
        return path
    return root / relative


def _cohort(name, config, env_var, public, extra=()):
    return Cohort(
        name=name,
        config=config,
        env_var=env_var,
        public=public,
        requires=_PLINK + tuple(extra),
    )


COHORTS = {
    c.name: c
    for c in (
        _cohort(
            "hgdp",
            "examples/hgdp_1kgp/config.yaml",
            "MG_HGDP_DATA",
            public=True,
            extra=("data/hgdp_project_labels.csv", "data/hgdp_project_geographic.csv"),
        ),
        _cohort(
            "ukbb_projection",
            "examples/ukbb/hgdp_1kgp_proj/config.yaml",
            "MG_UKBB_PROJECTION_DATA",
            public=False,
            extra=("data/fit_labels.csv", "data/project_labels.csv"),
        ),
        _cohort(
            "ukbb_subsample",
            "examples/ukbb/10k_WB_5K_Irish/config.yaml",
            "MG_UKBB_SUBSAMPLE_DATA",
            public=False,
            extra=("data/fit_labels.csv", "data/project_labels.csv"),
        ),
        _cohort(
            "ukbb_geosketch",
            "examples/ukbb/geosketch_phate/config.yaml",
            "MG_UKBB_GEOSKETCH_DATA",
            public=False,
            extra=("data/fit_labels.csv", "data/project_labels.csv"),
        ),
        _cohort(
            "aou_projection",
            "examples/aou/hgdp_1kgp_proj/config.yaml",
            "MG_AOU_PROJECTION_DATA",
            public=False,
            extra=("data/fit_labels.csv", "data/project_labels.csv"),
        ),
        _cohort(
            "aou_subsample",
            "examples/aou/10k_WBH/config.yaml",
            "MG_AOU_SUBSAMPLE_DATA",
            public=False,
            extra=("data/fit_labels.csv", "data/project_labels.csv"),
        ),
        _cohort(
            "aou_geosketch",
            "examples/aou/geosketch_phate/config.yaml",
            "MG_AOU_GEOSKETCH_DATA",
            public=False,
            extra=("data/fit_labels.csv", "data/project_labels.csv"),
        ),
    )
}
