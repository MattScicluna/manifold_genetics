"""
popVAE: a variational autoencoder fitted to genotypes (Battey, Coffing & Kern, G3 2021).

Runs the authors' ``popvae.py`` (https://github.com/kr-colab/popvae) as a
subprocess, because it pins Python 3.10 / TensorFlow 2.15. Point the wrapper at
that environment with ``python=`` / ``$POPVAE_PYTHON`` and at the script with
``script=`` / ``$POPVAE_SCRIPT``.

Unlike the other methods, popVAE embeds *genotypes*, not PCA coordinates. The
input passed to ``fit`` only fixes which samples are embedded and their order;
the genotypes come from the PLINK fileset given as ``genotypes``. The PLINK
genotypes are written to popVAE's own ``.popvae.hdf5`` input format, applying
the same preprocessing popVAE applies to a VCF: singletons dropped, missing calls
filled with Binomial(2, allele frequency) draws, dosages scaled to 0/0.5/1.
popVAE has no out-of-sample projection, so ``transform`` only accepts the fitted
samples.
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


def plink_to_popvae_hdf5(prefix: Union[str, Path], sample_ids, out_path: Union[str, Path], seed: int = 0):
    """Write ``sample_ids`` from PLINK ``prefix`` as a popVAE ``.popvae.hdf5`` file."""
    try:
        import h5py
        from bed_reader import open_bed
    except ImportError as exc:
        raise ImportError(
            "popVAE input conversion needs `bed-reader` and `h5py`: "
            "`pip install 'manifold-genetics[popvae]'`."
        ) from exc

    prefix = str(prefix)
    with open_bed(prefix + ".bed") as bed:
        iid = np.asarray(bed.iid).astype(str)
        pos = pd.Series(np.arange(len(iid)), index=iid)
        missing = [s for s in sample_ids if s not in pos.index]
        if missing:
            raise ValueError(f"{len(missing)} samples not in {prefix}.fam, e.g. {missing[:3]}")
        G = bed.read(index=np.s_[pos.loc[list(sample_ids)].values, :], dtype="float32")

    # popVAE preprocessing (popvae.py, VCF branch): drop singletons, impute, scale.
    miss = np.isnan(G)
    dc_all = np.nansum(G, axis=0)
    ninds = (~miss).sum(axis=0)
    keep = dc_all > 2
    G, miss, dc_all, ninds = G[:, keep], miss[:, keep], dc_all[keep], ninds[keep]
    af = dc_all / (2 * np.maximum(ninds, 1))
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(miss)
    G[rows, cols] = rng.binomial(2, af[cols])
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("derived_counts", data=(G * 0.5).astype("float32"))
        h5.create_dataset("samples", data=np.asarray(sample_ids, dtype=object), dtype=h5py.string_dtype())
    logger.info(f"Wrote popVAE input: {G.shape[0]} samples x {G.shape[1]} SNPs -> {out_path}")
    return out_path


class PopVAE(EmbeddingBase):
    """popVAE embedding (latent means).

    Args:
        genotypes: PLINK prefix (``.bed/.bim/.fam``) holding the samples to embed.
        n_components: Latent dimensions (``--latent_dim``).
        python: Interpreter of the popVAE environment (default ``$POPVAE_PYTHON``).
        script: Path to ``popvae.py`` (default ``$POPVAE_SCRIPT``).
        patience, max_epochs, batch_size, train_prop, depth, width: popVAE options
            (popVAE defaults: 50, 500, 32, 0.9, 6, 128).
        search_network_sizes: Run popVAE's own grid search over width/depth.
        max_snps: Randomly subsample SNPs (``--max_SNPs``); ``None`` uses all.
        workdir: Where inputs/outputs go (default: a temporary directory).
        random_state: Seed (``--seed``).
    """

    def __init__(
        self,
        genotypes: Optional[Union[str, Path]] = None,
        n_components: int = 2,
        python: Optional[str] = None,
        script: Optional[str] = None,
        patience: int = 50,
        max_epochs: int = 500,
        batch_size: int = 32,
        train_prop: float = 0.9,
        depth: int = 6,
        width: int = 128,
        search_network_sizes: bool = False,
        max_snps: Optional[int] = None,
        workdir: Optional[Union[str, Path]] = None,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.genotypes = genotypes
        self.python = python or os.environ.get("POPVAE_PYTHON") or sys.executable
        self.script = script or os.environ.get("POPVAE_SCRIPT")
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.train_prop = train_prop
        self.depth = depth
        self.width = width
        self.search_network_sizes = search_network_sizes
        self.max_snps = max_snps
        self.workdir = workdir

    def _command(self, infile: Path, out: Path) -> list:
        if not self.script:
            raise ValueError("Set `script=` or $POPVAE_SCRIPT to the authors' popvae.py")
        cmd = [
            self.python, self.script,
            "--infile", str(infile), "--out", str(out),
            "--latent_dim", str(self.n_components),
            "--patience", str(self.patience), "--max_epochs", str(self.max_epochs),
            "--batch_size", str(self.batch_size), "--train_prop", str(self.train_prop),
            "--depth", str(self.depth), "--width", str(self.width),
        ]
        if self.random_state is not None:
            cmd += ["--seed", str(self.random_state)]
        if self.search_network_sizes:
            cmd += ["--search_network_sizes"]
        if self.max_snps:
            cmd += ["--max_SNPs", str(self.max_snps)]
        return cmd

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "PopVAE":
        if self.genotypes is None:
            raise ValueError("popVAE embeds genotypes: pass `genotypes=` (a PLINK prefix)")
        _, sample_ids = self._load_input_data(X)
        sample_ids = [str(s) for s in sample_ids]
        workdir = Path(self.workdir or tempfile.mkdtemp(prefix="popvae_"))
        workdir.mkdir(parents=True, exist_ok=True)
        infile = workdir / "input.popvae.hdf5"
        if not infile.exists():
            plink_to_popvae_hdf5(self.genotypes, sample_ids, infile, seed=self.random_state or 0)
        out = workdir / f"popvae_seed{self.random_state}"
        cmd = self._command(infile, out)
        logger.info("Running popVAE: " + " ".join(shlex.quote(c) for c in cmd))
        with open(str(out) + ".log", "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=workdir)
        coords = pd.read_csv(str(out) + "_latent_coords.txt", sep="\t")
        coords["sampleID"] = coords["sampleID"].astype(str)
        coords = coords.set_index("sampleID").loc[sample_ids]
        cols = [f"LD{i + 1}" for i in range(self.n_components)]
        self._embedding = coords[cols].values
        self._sample_ids = sample_ids
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("PopVAE must be fitted before transform")
        _, sample_ids = self._load_input_data(X)
        if [str(s) for s in sample_ids] != self._sample_ids:
            raise NotImplementedError("popVAE has no out-of-sample projection; embed all samples in fit")
        return self._format_output(self._embedding, self._sample_ids)

    def fit_transform(self, X, output_path=None) -> pd.DataFrame:
        self.fit(X)
        return self._format_output(self._embedding, self._sample_ids, output_path)
