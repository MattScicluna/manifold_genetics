"""Read and write flashpca's artefact set.

The pipeline's output layout is a contract, and it should not depend on which
backend produced it. So the Python backend writes exactly what flashpca writes
rather than a private format. Two consequences beyond a green contract test:

* a model fitted by either backend can be projected by the other -- flashpca
  reads ``--inload``/``--inmeansd``, and :func:`read_model` reads the same files;
* the artefacts stay human-inspectable, which a ``.npz`` is not.

Formats, taken from real flashpca output rather than documentation::

    <prefix>.meansd    SNP  RefAllele  Mean  SD      tab-separated, header
    <prefix>.loadings  SNP  RefAllele  V1..Vd        tab-separated, header
    <prefix>.eigenval  one value per line            no header
    <prefix>.eigenvec  FID  IID  U1..Ud              tab-separated, header
    <prefix>.PC        FID  IID  PC1..PCd            tab-separated, header
    pve.txt            one proportion per line       no header, beside the prefix

``.eigenvec`` holds U with unit-norm columns; ``.PC`` holds ``U * sqrt(eigenvalues)``.
Confusing the two silently rescales every downstream coordinate.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .backends.base import PCAModel

__all__ = ["model_files_exist", "read_model", "write_model"]

PathLike = Union[str, Path]

# flashpca prints about 7 significant digits; a little more costs nothing and
# makes the round trip lossless enough to reuse as a checkpoint.
_FLOAT_FMT = "%.10g"


def _paths(prefix: PathLike) -> dict:
    prefix = Path(prefix)
    return {
        "meansd": prefix.with_suffix(prefix.suffix + ".meansd"),
        "loadings": prefix.with_suffix(prefix.suffix + ".loadings"),
        "eigenval": prefix.with_suffix(prefix.suffix + ".eigenval"),
        "eigenvec": prefix.with_suffix(prefix.suffix + ".eigenvec"),
        "pc": prefix.with_suffix(prefix.suffix + ".PC"),
        "pve": prefix.parent / "pve.txt",
    }


def model_files_exist(prefix: PathLike) -> bool:
    """True when the files :func:`read_model` needs are all present."""
    p = _paths(prefix)
    return all(p[k].exists() for k in ("meansd", "loadings", "eigenval"))


def write_model(model: PCAModel, prefix: PathLike) -> dict:
    """Write ``model`` as flashpca's artefact set. Returns the paths written."""
    paths = _paths(prefix)
    paths["meansd"].parent.mkdir(parents=True, exist_ok=True)

    n_components = model.n_components

    pd.DataFrame(
        {
            "SNP": list(model.variant_ids),
            "RefAllele": list(model.ref_alleles),
            "Mean": model.mean,
            "SD": model.sd,
        }
    ).to_csv(paths["meansd"], sep="\t", index=False, float_format=_FLOAT_FMT)

    loadings = pd.DataFrame(model.loadings, columns=[f"V{i + 1}" for i in range(n_components)])
    loadings.insert(0, "RefAllele", list(model.ref_alleles))
    loadings.insert(0, "SNP", list(model.variant_ids))
    loadings.to_csv(paths["loadings"], sep="\t", index=False, float_format=_FLOAT_FMT)

    np.savetxt(paths["eigenval"], model.eigenvalues, fmt=_FLOAT_FMT)

    # flashpca's pve is each eigenvalue over the total variance. Without a
    # recorded total the eigenvalues are all we have, so the proportions are
    # relative to the components retained -- still monotone, still in (0, 1].
    total = model.total_variance or float(np.sum(model.eigenvalues))
    np.savetxt(paths["pve"], np.asarray(model.eigenvalues) / total, fmt=_FLOAT_FMT)

    if model.fit_coords is not None:
        iids = list(model.fit_sample_ids or [])
        fids = list(model.fit_family_ids or iids)
        scale = np.sqrt(model.eigenvalues)
        with np.errstate(divide="ignore", invalid="ignore"):
            U = np.divide(
                model.fit_coords, scale, out=np.zeros_like(model.fit_coords), where=scale > 0
            )

        for path, block, tag in (
            (paths["eigenvec"], U, "U"),
            (paths["pc"], model.fit_coords, "PC"),
        ):
            frame = pd.DataFrame(block, columns=[f"{tag}{i + 1}" for i in range(n_components)])
            frame.insert(0, "IID", iids)
            frame.insert(0, "FID", fids)
            frame.to_csv(path, sep="\t", index=False, float_format=_FLOAT_FMT)

    return paths


def read_model(prefix: PathLike) -> PCAModel:
    """Read a model written by :func:`write_model` or by flashpca itself.

    Raises:
        FileNotFoundError: the .meansd, .loadings or .eigenval file is missing.
    """
    paths = _paths(prefix)
    for key in ("meansd", "loadings", "eigenval"):
        if not paths[key].exists():
            raise FileNotFoundError(f"No PCA model at {prefix}: {paths[key]} is missing")

    meansd = pd.read_csv(paths["meansd"], sep="\t")
    loadings = pd.read_csv(paths["loadings"], sep="\t")
    eigenvalues = np.loadtxt(paths["eigenval"], ndmin=1)

    fit_coords: Optional[np.ndarray] = None
    fit_sample_ids = None
    fit_family_ids = None
    if paths["pc"].exists():
        pc = pd.read_csv(paths["pc"], sep="\t")
        fit_coords = pc.filter(regex=r"^PC\d+$").to_numpy()
        fit_sample_ids = [str(v) for v in pc["IID"]]
        fit_family_ids = [str(v) for v in pc["FID"]]

    total_variance = None
    if paths["pve"].exists():
        pve = np.loadtxt(paths["pve"], ndmin=1)
        if pve.size and pve[0] > 0:
            total_variance = float(eigenvalues[0] / pve[0])

    return PCAModel(
        mean=meansd["Mean"].to_numpy(),
        sd=meansd["SD"].to_numpy(),
        loadings=loadings.filter(regex=r"^V\d+$").to_numpy(),
        eigenvalues=eigenvalues,
        variant_ids=[str(v) for v in meansd["SNP"]],
        ref_alleles=[str(v) for v in meansd["RefAllele"]],
        fit_coords=fit_coords,
        fit_sample_ids=fit_sample_ids,
        fit_family_ids=fit_family_ids,
        total_variance=total_variance,
    )
