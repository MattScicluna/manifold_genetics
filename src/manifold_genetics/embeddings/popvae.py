"""
popVAE: a variational autoencoder fitted to genotypes (Battey, Coffing & Kern, G3 2021).

Runs the authors' ``popvae.py`` (https://github.com/kr-colab/popvae) as a
subprocess, because it pins Python 3.10 / TensorFlow 2.15. Point the wrapper at
that environment with ``python=`` / ``$POPVAE_PYTHON`` and at the script with
``script=`` / ``$POPVAE_SCRIPT``.

Unlike the other methods, popVAE embeds *genotypes*, not PCA coordinates. The
input passed to ``fit`` only fixes which samples are embedded and their order;
the genotypes come from the PLINK fileset given as ``genotypes``. They are
written as a scikit-allel style zarr store (``calldata/GT``, ``samples``), one
of popVAE's documented input formats, so popVAE applies its own filtering
(biallelic sites, singletons) and missing-data imputation. The conversion runs
in the popVAE environment, which needs ``bed-reader`` installed. (popVAE's
``.popvae.hdf5`` input is not used: with h5py 3 its sample IDs load as bytes and
popvae.py fails on them.) popVAE has no out-of-sample projection, so
``transform`` only accepts the fitted samples.
"""

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import EmbeddingBase

logger = logging.getLogger(__name__)


_CONVERT = r"""
import sys
import numcodecs
import numpy as np
import zarr
from bed_reader import open_bed

prefix, ids_file, out = sys.argv[1:4]
ids = [l.strip() for l in open(ids_file) if l.strip()]
with open_bed(prefix + ".bed") as bed:
    iid = np.asarray(bed.iid).astype(str)
    pos = {s: i for i, s in enumerate(iid)}
    missing = [s for s in ids if s not in pos]
    if missing:
        sys.exit(f"{len(missing)} samples not in {prefix}.fam, e.g. {missing[:3]}")
    rows = np.array([pos[s] for s in ids])
    n_var = bed.sid_count
    root = zarr.open_group(out, mode="w")
    # Object dtype, as scikit-allel's read_vcf returns: popvae.py concatenates a
    # str with this array, which fails for fixed-width unicode arrays.
    root.create_dataset("samples", data=np.array(ids, dtype=object), object_codec=numcodecs.VLenUTF8())
    gt = root.require_group("calldata").zeros(
        "GT", shape=(n_var, len(ids), 2), chunks=(10000, len(ids), 2), dtype="i1"
    )
    for start in range(0, n_var, 10000):
        d = bed.read(index=np.s_[rows, start:start + 10000], dtype="float32").T  # variants x samples
        g = np.stack([d >= 1, d >= 2], axis=-1).astype("i1")  # allele-1 count -> two calls
        g[np.isnan(d)] = -1
        gt[start:start + d.shape[0]] = g
print(f"wrote {n_var} variants x {len(ids)} samples to {out}")
"""


def plink_to_popvae_zarr(prefix, sample_ids, out_path, python: str):
    """Write ``sample_ids`` from PLINK ``prefix`` as a zarr store popVAE can read.

    Runs in the popVAE interpreter ``python`` (its zarr version matches popVAE's
    reader; it needs ``bed-reader``). Genotypes are encoded as counts of the
    PLINK A1 allele.
    """
    out_path = Path(out_path)
    ids_file = out_path.with_suffix(".samples.txt")
    ids_file.write_text("\n".join(str(s) for s in sample_ids) + "\n")
    script = out_path.with_suffix(".convert.py")
    script.write_text(_CONVERT)
    subprocess.run([python, str(script), str(prefix), str(ids_file), str(out_path)], check=True)
    return out_path


class PopVAE(EmbeddingBase):
    """popVAE embedding (latent means).

    popVAE's own training defaults (network size, patience, epochs, train/validation
    split) are used unchanged.

    Args:
        genotypes: PLINK prefix (``.bed/.bim/.fam``) holding the samples to embed.
        n_components: Latent dimensions (``--latent_dim``).
        python: Interpreter of the popVAE environment (default ``$POPVAE_PYTHON``).
        script: Path to ``popvae.py`` (default ``$POPVAE_SCRIPT``).
        workdir: Where inputs/outputs go (default: a temporary directory). The
            converted genotypes are reused across fits in the same ``workdir``.
        random_state: Seed (``--seed``).
    """

    def __init__(
        self,
        genotypes: Optional[Union[str, Path]] = None,
        n_components: int = 2,
        python: Optional[str] = None,
        script: Optional[str] = None,
        workdir: Optional[Union[str, Path]] = None,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.genotypes = genotypes
        self.python = python or os.environ.get("POPVAE_PYTHON") or sys.executable
        self.script = script or os.environ.get("POPVAE_SCRIPT")
        self.workdir = workdir

    def _command(self, infile: Path, out: Path) -> list:
        if not self.script:
            raise ValueError("Set `script=` or $POPVAE_SCRIPT to the authors' popvae.py")
        cmd = [self.python, self.script, "--infile", str(infile), "--out", str(out)]
        cmd += ["--latent_dim", str(self.n_components)]
        if self.random_state is not None:
            cmd += ["--seed", str(self.random_state)]
        return cmd

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "PopVAE":
        if self.genotypes is None:
            raise ValueError("popVAE embeds genotypes: pass `genotypes=` (a PLINK prefix)")
        _, sample_ids = self._load_input_data(X)
        sample_ids = [str(s) for s in sample_ids]
        workdir = Path(self.workdir or tempfile.mkdtemp(prefix="popvae_"))
        workdir.mkdir(parents=True, exist_ok=True)
        infile = workdir / "input.zarr"
        if not infile.exists():
            plink_to_popvae_zarr(self.genotypes, sample_ids, infile, python=self.python)
        out = workdir / f"popvae_seed{self.random_state}"
        cmd = self._command(infile, out)
        logger.info("Running popVAE: " + " ".join(shlex.quote(c) for c in cmd))
        with open(str(out) + ".log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=workdir)
        coords = pd.read_csv(str(out) + "_latent_coords.txt", sep="\t")
        coords["sampleID"] = coords["sampleID"].astype(str)
        coords = coords.set_index("sampleID").loc[sample_ids]
        # popvae.py names the latent means mean1/mean2 when latent_dim == 2, LD1..LDn otherwise.
        cols = (
            ["mean1", "mean2"]
            if self.n_components == 2
            else [f"LD{i + 1}" for i in range(self.n_components)]
        )
        self._embedding = coords[cols].values
        self._sample_ids = sample_ids
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("PopVAE must be fitted before transform")
        _, sample_ids = self._load_input_data(X)
        if [str(s) for s in sample_ids] != self._sample_ids:
            raise NotImplementedError(
                "popVAE has no out-of-sample projection; embed all samples in fit"
            )
        return self._format_output(self._embedding, self._sample_ids)

    def fit_transform(self, X, output_path=None) -> pd.DataFrame:
        self.fit(X)
        return self._format_output(self._embedding, self._sample_ids, output_path)
