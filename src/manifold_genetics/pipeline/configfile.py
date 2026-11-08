"""Load a pipeline run from a YAML config file.

``examples/_shared/run_pipeline.sh`` is ~400 lines re-declaring an argument
surface ``cli.py`` already owns, and that duplication is where this repo's drift
keeps coming from: a subsample default every caller had to correct, one script
silently taking the most expensive landmarking path. A config file is a
serialisation of :func:`~manifold_genetics.pipeline.config.build_configs`'
keyword arguments, so the shell layer stops existing rather than being rewritten.

A config looks like::

    preset: subsample          # optional; supplies the mode's defaults

    data:
      fit_plink: data/fit_subset
      project_plink: data/project_subset
      labels: data/labels.csv          # or fit_labels + project_labels
      colormap: ../colormaps/ukbb.json
      output_dir: outputs

    pca:
      n_pcs: 20

    embedding:
      method: phate
      t: 100                   # overrides the preset

    skip:
      admixture: true

    painting:                  # optional; replaces genotype PCA as the embedding input
      enabled: true            # needs data.phased_vcf: data/chr{chr}.vcf.gz
      method: hc               # or finestructure
      missing: error           # or drop: leave out cohort samples the VCFs lack
      pbwt_dir: shared/pbwt    # optional; share pbwt's output between runs

Two decisions worth knowing:

* **Paths resolve relative to the config file**, not the working directory, so an
  example runs from anywhere.
* **Unknown keys are errors, not warnings.** Silently ignoring a typo is how a
  setting ends up not doing what the file says it does.
"""

import difflib
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

from .config import check_landmarking, validate_method_grid, validate_phate_grid

__all__ = [
    "ConfigFileError",
    "PRESETS",
    "PRESET_ALIASES",
    "PRESET_OWNED_EMBEDDING_KEYS",
    "load_config",
]

PathLike = Union[str, Path]


class ConfigFileError(ValueError):
    """A config file is missing, malformed, or contains unknown keys."""


# The mode policy that used to live in examples/_shared/run_pipeline.sh, with the
# landmarking rules settled in PR #82. n_landmark and random_landmarking are set
# together: n_landmark alone silently selects the far more expensive spectral
# path, which is exactly the bug that policy fixed.
PRESETS: Dict[str, Dict[str, Any]] = {
    "projection": {
        "embedding_input": "both",
        "embedding": {"knn": 100, "t": 3, "n_landmark": None, "random_landmarking": False},
    },
    "subsample": {
        "embedding_input": "fit",
        "embedding": {
            "knn": 500,
            "t": 50,
            "n_landmark": 10000,
            "random_landmarking": True,
        },
    },
    "whole_cohort": {
        "embedding_input": "project",
        "embedding": {"knn": 100, "t": 3, "n_landmark": None, "random_landmarking": False},
    },
}

# Every embedding key a preset sets, derived from PRESETS rather than typed out
# separately so the two cannot drift. A carried `embedding` section that still
# names one of these -- or `input_mode` -- after a preset change would silently
# override what the new preset sets; `cohort.write_cohort_config` drops them.
PRESET_OWNED_EMBEDDING_KEYS = frozenset(
    key for spec in PRESETS.values() for key in spec.get("embedding", {})
)

# Renamed 2026-09-12. This project used `transform` for two different things: the
# second cohort's dataset role, and the sklearn-style method verb. The role became
# `project`; this preset was the last holdout, denoting a *mode* while every other
# use of the word denotes an *operation*. The rule now: `transform` is the verb,
# `project` is the dataset role, and a preset is named for the shape of the run.
#
# Accepted but not advertised -- an alias that appears in the error listing is a
# name people keep choosing, which defeats renaming it.
PRESET_ALIASES: Dict[str, str] = {"transform": "whole_cohort"}

# section -> {key in file: keyword argument of run_pipeline}
_DATA_KEYS = {
    "fit_plink": "fit_plink",
    "project_plink": "project_plink",
    "output_dir": "output_dir",
    "labels": "labels",
    "fit_labels": "fit_labels",
    "project_labels": "project_labels",
    "colormap": "colormap",
    "fit_colormap": "fit_colormap",
    "project_colormap": "project_colormap",
    "geographic_coords": "geographic_coords",
    # One phased VCF per chromosome, as a path containing "{chr}". Read only
    # when the painting section enables painting.
    "phased_vcf": "phased_vcf",
}
_PATH_KEYS = set(_DATA_KEYS)

_PCA_KEYS = {
    "n_pcs": "n_pcs",
    "backend": "pca_backend",
    "force": "force_pca",
    # Memory budgets, in GB. Worth setting from the config because the right
    # value is a property of the machine the run is submitted to, not of the
    # cohort -- and because the fit silently streams at ~19x the wall clock
    # above the budget, which is the kind of slowness nobody diagnoses.
    "max_fit_memory_gb": "max_fit_memory_gb",
    "max_project_memory_gb": "max_project_memory_gb",
}
_ADMIXTURE_KEYS = {
    "k_min": "k_min",
    "k_max": "k_max",
    "threads": "admix_threads",
    "num_gpus": "admix_gpus",
    "batch_size": "admix_batch_size",
}
_PAINTING_KEYS = {
    "enabled": "painting",
    "method": "painting_method",
    "chromosomes": "painting_chromosomes",
    "force": "force_painting",
    "missing": "painting_missing",
    "pbwt_dir": "painting_pbwt_dir",
}
# Resolved against the config file's directory, like the data section's paths.
_PAINTING_PATH_KEYS = {"pbwt_dir"}
_VIZ_KEYS = {
    "admix_group_column": "admix_group_column",
    "admix_within_group_order": "admix_within_group_order",
    "projection_plot_fit_column": "projection_plot_fit_column",
    "projection_plot_project_column": "projection_plot_project_column",
    # Comparison figures, validated by load_config and build_configs.
    "method_grid": "method_grid",
    "phate_grid": "phate_grid",
}
_SKIP_KEYS = {
    "pca": "skip_pca",
    "admixture": "skip_admixture",
    "embedding": "skip_embedding",
    "pca_visualization": "skip_pca_visualization",
    "visualization": "skip_visualization",
    "admixture_visualization": "skip_admixture_visualization",
    "metrics": "skip_metrics",
}
# `method` and `input_mode` are run_pipeline arguments; everything else in the
# embedding section is a method-specific parameter passed through as a dict.
_EMBEDDING_ARGS = {"method": "embedding", "input_mode": "embedding_input"}
_EMBEDDING_PARAMS = {
    # Applies to every method, unlike the rest of this set.
    "n_components",
    "knn",
    "t",
    "n_landmark",
    "random_landmarking",
    "embed_batch_size",
    "gamma",
    "n_neighbors",
    "min_dist",
    "perplexity",
}

_SECTIONS = {
    "preset",
    "data",
    "pca",
    "painting",
    "admixture",
    "embedding",
    "visualization",
    "skip",
}


def _reject_unknown(given, allowed, where: str) -> None:
    unknown = sorted(set(given) - set(allowed))
    if not unknown:
        return
    lines = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(allowed), n=1, cutoff=0.6)
        lines.append(f"  {key!r}" + (f" -- did you mean {close[0]!r}?" if close else ""))
    raise ConfigFileError(
        f"Unknown {where}:\n" + "\n".join(lines) + f"\n\nValid: {', '.join(sorted(allowed))}"
    )


def _resolve(value, base: Path):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_yaml(path: Path) -> Mapping:
    if not path.exists():
        raise ConfigFileError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"Could not parse YAML in {path}:\n{exc}")
    if not isinstance(data, Mapping):
        raise ConfigFileError(
            f"{path} does not contain a mapping. A config needs at least a 'data' section."
        )
    return data


def load_config(path: PathLike, base_dir: Optional[PathLike] = None) -> Dict[str, Any]:
    """Read ``path`` and return keyword arguments for ``run_pipeline``.

    Args:
        path: The YAML config file.
        base_dir: Directory relative paths resolve against. Defaults to the
            config file's own directory, which is what makes an example runnable
            from anywhere.

    Raises:
        ConfigFileError: missing file, malformed YAML, unknown section or key,
            unknown preset, or a contradictory embedding setting.
    """
    path = Path(path).expanduser()
    data = _load_yaml(path)
    base = Path(base_dir) if base_dir is not None else path.parent.resolve()

    _reject_unknown(data, _SECTIONS, f"section(s) in {path.name}")

    preset_name = data.get("preset")
    if preset_name in PRESET_ALIASES:
        replacement = PRESET_ALIASES[preset_name]
        warnings.warn(
            f"The preset {preset_name!r} has been renamed to {replacement!r}, because "
            f"{preset_name!r} is the name of a method rather than of a mode. The old "
            "name still works and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        preset_name = replacement
    if preset_name is not None and preset_name not in PRESETS:
        raise ConfigFileError(
            f"Unknown preset {preset_name!r}. Choose from: {', '.join(sorted(PRESETS))}"
        )
    preset = PRESETS.get(preset_name, {})

    if "data" not in data:
        raise ConfigFileError(
            f"{path.name} has no 'data' section. It must name at least fit_plink, "
            "project_plink, labels, colormap and output_dir."
        )

    kwargs: Dict[str, Any] = {}

    section = data["data"] or {}
    _reject_unknown(section, _DATA_KEYS, "key(s) in the 'data' section")
    for key, value in section.items():
        kwargs[_DATA_KEYS[key]] = _resolve(value, base) if key in _PATH_KEYS else value

    for name, mapping in (
        ("pca", _PCA_KEYS),
        ("painting", _PAINTING_KEYS),
        ("admixture", _ADMIXTURE_KEYS),
        ("visualization", _VIZ_KEYS),
        ("skip", _SKIP_KEYS),
    ):
        section = data.get(name) or {}
        _reject_unknown(section, mapping, f"key(s) in the {name!r} section")
        for key, value in section.items():
            if name == "painting" and key in _PAINTING_PATH_KEYS and value is not None:
                value = _resolve(value, base)
            kwargs[mapping[key]] = value

    for key, arg in _SKIP_KEYS.items():
        kwargs.setdefault(arg, False)

    # --- embedding: preset defaults, then the file's own values ---
    for key, value in preset.items():
        if key != "embedding":
            kwargs.setdefault(key, value)
    params: Dict[str, Any] = dict(preset.get("embedding", {}))

    section = data.get("embedding") or {}
    _reject_unknown(
        section, set(_EMBEDDING_ARGS) | _EMBEDDING_PARAMS, "key(s) in the 'embedding' section"
    )
    for key, value in section.items():
        if key in _EMBEDDING_ARGS:
            kwargs[_EMBEDDING_ARGS[key]] = value
        else:
            params[key] = value

    _normalise_embedding_params(params)
    try:
        check_landmarking(params)
    except ValueError as exc:
        raise ConfigFileError(str(exc)) from None

    # The grids' parameters are embedding parameters too: the same spelling of
    # none, and checked here, against the run's own settings they inherit, so a
    # mistake is reported when the file is read rather than after PCA.
    method_grid, phate_grid = kwargs.get("method_grid"), kwargs.get("phate_grid")
    try:
        if isinstance(method_grid, Mapping):
            method_grid = {
                method: _normalise_embedding_params(dict(p)) if isinstance(p, Mapping) else p
                for method, p in method_grid.items()
            }
        if method_grid is not None:
            validate_method_grid(method_grid, params)
            kwargs["method_grid"] = method_grid
        if isinstance(phate_grid, Mapping):
            phate_grid = _normalise_embedding_params(dict(phate_grid))
        if phate_grid is not None:
            validate_phate_grid(phate_grid, params)
            kwargs["phate_grid"] = phate_grid
    except ValueError as exc:
        raise ConfigFileError(str(exc)) from None

    kwargs["embedding_params"] = params
    return kwargs


def _normalise_n_landmark(value):
    if isinstance(value, str):
        return None if value.lower() == "none" else int(value)
    return value


def _normalise_embedding_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Accept YAML's ``n_landmark: none``, which is a string, as None.

    Also inside a ``phate_grid`` list (``n_landmark: [none, 2000]``). Changes
    ``params`` in place and returns it.
    """
    if "n_landmark" in params:
        value = params["n_landmark"]
        params["n_landmark"] = (
            [_normalise_n_landmark(v) for v in value]
            if isinstance(value, list)
            else _normalise_n_landmark(value)
        )
    return params
