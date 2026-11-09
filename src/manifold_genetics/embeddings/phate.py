"""
PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding).

Wrapper for the PHATE algorithm with a consistent API.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import phate

from .base import EmbeddingBase

logger = logging.getLogger(__name__)


class PHATE(EmbeddingBase):
    """
    PHATE embedding for manifold learning.

    Examples:
        >>> # Basic usage
        >>> phate_model = PHATE(n_components=2, knn=25)
        >>> embedding = phate_model.fit_transform("pca_50.csv")
        >>>
        >>> # With custom parameters
        >>> phate_model = PHATE(n_components=2, knn=50, t=15, gamma=1)
        >>> embedding = phate_model.fit_transform("pca_50.csv", output_path="phate_2d.csv")
    """

    def __init__(
        self,
        n_components: int = 2,
        knn: int = 5,
        t: Union[int, str] = "auto",
        decay: Optional[int] = 40,
        gamma: float = 1.0,
        n_pca: Optional[int] = 100,
        n_landmark: Optional[int] = 2000,
        random_landmarking: bool = False,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        mds_solver: str = "sgd",
        embed_batch_size: Optional[int] = None,
    ):
        """
        Initialize PHATE embedding.

        Args:
            n_components: Number of embedding dimensions
            knn: Number of nearest neighbors
            t: Diffusion time (int or 'auto')
            decay: Alpha decay parameter for graph kernel
            gamma: Informational distance parameter
            n_pca: Number of PCs to compute before PHATE (None to skip)
            n_landmark: Number of landmarks for efficiency
            random_landmarking: Use random landmark selection instead of spectral clustering
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
            mds_solver: MDS solver ('sgd' or 'smacof')
            embed_batch_size: Number of samples to transform at once (None for all)
        """
        super().__init__(n_components=n_components, random_state=random_state)

        self.knn = knn
        self.t = t
        self.decay = decay
        self.gamma = gamma
        self.n_pca = n_pca
        self.n_landmark = n_landmark
        self.random_landmarking = random_landmarking
        self.n_jobs = n_jobs
        self.mds_solver = mds_solver
        self.embed_batch_size = embed_batch_size

        # Initialize PHATE model
        self.model = phate.PHATE(
            n_components=n_components,
            knn=knn,
            t=t,
            decay=decay,
            gamma=gamma,
            n_pca=n_pca,
            n_landmark=n_landmark,
            random_landmarking=random_landmarking,
            random_state=random_state,
            n_jobs=n_jobs,
            mds_solver=mds_solver,
            verbose=0,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "PHATE":
        """
        Fit PHATE on the data.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            Self (for method chaining)
        """
        X_array, self._sample_ids = self._load_input_data(X)
        self._check_exact_fits_in_memory(len(X_array))

        logger.info(f"Fitting PHATE with knn={self.knn}, t={self.t}...")
        self.model.fit(X_array)
        # Kept to recognise the fit data in transform(); PHATE holds it already.
        self._fit_X = X_array
        self._is_fitted = True

        logger.info(f"✓ PHATE fitted on {len(X_array)} samples")
        return self

    # Without landmarks PHATE holds dense n x n operators. Measured on UKBB
    # (n = 59,264; one n x n is 26.2 GB): the fit peaked at ~1x, the transform
    # passed 2.2x before the 56 GB ceiling stopped it, and a synthetic run held
    # 3x. So 3 x 8 n^2 bytes is the estimate.
    EXACT_DENSE_COPIES = 3

    def _check_exact_fits_in_memory(self, n_samples: int) -> None:
        """Refuse an exact (no-landmark) run that cannot fit, before it starts.

        Exact is the default and stays so -- it is what every result so far was
        computed with -- but at biobank scale it needs hundreds of GB and is
        OOM-killed tens of minutes in, after the fit has already succeeded.
        Failing here, with the remedy, costs nothing. MANIFOLD_GENETICS_FORCE_EXACT=1
        skips the check.
        """
        if self.n_landmark is not None or os.environ.get("MANIFOLD_GENETICS_FORCE_EXACT"):
            return
        from ..utils.memory import available_memory_bytes

        needed = self.EXACT_DENSE_COPIES * 8 * n_samples**2
        available = available_memory_bytes()
        if available is None or needed <= available:
            return
        raise MemoryError(
            f"PHATE without landmarks on {n_samples:,} samples needs about "
            f"{needed / 1024**3:,.0f} GB (dense {n_samples:,} x {n_samples:,} operators), "
            f"but about {available / 1024**3:,.0f} GB is available. Use landmarks, as the "
            "subsample preset does: --n-landmark 10000 --random-landmarking (or fewer "
            "landmarks for less memory). Set MANIFOLD_GENETICS_FORCE_EXACT=1 to run exact anyway."
        )

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        Transform data using fitted PHATE.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)

        Returns:
            DataFrame with sample_id and PHATE coordinates
        """
        if not self._is_fitted:
            raise RuntimeError("PHATE not fitted. Call fit() first.")

        X_array, sample_ids = self._load_input_data(X)
        n_samples = len(X_array)

        # The fit data is never batched. Whole, PHATE recognises it and returns
        # its embedding; sliced, each batch is "new data" and is re-extended
        # through a fresh neighbour search -- on UKBB (59,264 fit samples) the
        # same peak memory, ~15x the time, and approximate coordinates (#157).
        is_fit_data = self._is_fit_data(X_array)
        if is_fit_data and self.embed_batch_size is not None and n_samples > self.embed_batch_size:
            logger.info(
                f"Input is the fit data: returning PHATE's embedding without batching "
                f"(embed_batch_size={self.embed_batch_size} applies to new samples only)"
            )

        # Check if batch processing is needed
        if (
            not is_fit_data
            and self.embed_batch_size is not None
            and n_samples > self.embed_batch_size
        ):
            logger.info(
                f"🔄 BATCH MODE: Transforming {n_samples} samples with PHATE in batches of {self.embed_batch_size}..."
            )

            # Process in batches
            embeddings = []
            n_batches = int(np.ceil(n_samples / self.embed_batch_size))

            for i in range(n_batches):
                start_idx = i * self.embed_batch_size
                end_idx = min((i + 1) * self.embed_batch_size, n_samples)

                logger.info(
                    f"  Processing batch {i + 1}/{n_batches} (samples {start_idx}-{end_idx})..."
                )
                batch_data = X_array[start_idx:end_idx]
                batch_embedding = self.model.transform(batch_data)
                embeddings.append(batch_embedding)

            # Concatenate all batches
            embedding = np.vstack(embeddings)
            logger.info(f"✓ Batch transformation complete: {embedding.shape}")
        else:
            # Process all at once
            if self.embed_batch_size is not None:
                logger.warning(
                    f"embed_batch_size={self.embed_batch_size} set but not using batch mode "
                    f"(data size {n_samples} <= batch size)"
                )
            logger.info(f"Transforming {n_samples} samples with PHATE (no batching)...")
            embedding = self.model.transform(X_array)

        return self._format_output(embedding, sample_ids)

    def _is_fit_data(self, X_array: np.ndarray) -> bool:
        """Whether ``X_array`` is the matrix PHATE was fitted on."""
        fit_X = getattr(self, "_fit_X", None)
        return fit_X is not None and fit_X.shape == X_array.shape and np.array_equal(fit_X, X_array)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Fit PHATE and transform data in one step.

        Args:
            X: Input data (numpy array, DataFrame, or path to CSV)
            output_path: Optional path to save output CSV

        Returns:
            DataFrame with sample_id and PHATE coordinates
        """
        X_array, sample_ids = self._load_input_data(X)
        self._check_exact_fits_in_memory(len(X_array))

        logger.info(f"Running PHATE (knn={self.knn}, t={self.t}) on {len(X_array)} samples...")
        embedding = self.model.fit_transform(X_array)
        self._is_fitted = True
        self._sample_ids = sample_ids

        logger.info(f"✓ PHATE embedding complete: {embedding.shape}")

        # Format output with sample_id column
        return self._format_output(embedding, sample_ids, output_path)


def _check_genotype_weight(weight) -> None:
    if isinstance(weight, bool) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"The genotype weight must be in [0, 1], got {weight!r}")


def mix_kernels(K_genotype, K_painting, weight: float):
    """``weight * K_genotype / d_genotype + (1 - weight) * K_painting / d_painting``.

    ``d`` is a kernel's mean degree (its mean row sum), so both views enter on
    the same scale whatever the density of their neighborhoods, and a kernel
    keeps its shape. The sum is symmetrized, ``(K + K.T) / 2``, and returned as
    a sparse matrix. Weight 1 or 0 gives back one view's kernel up to a
    constant factor, which PHATE's row normalization removes.
    """
    from scipy import sparse

    K_g, K_p = sparse.csr_matrix(K_genotype), sparse.csr_matrix(K_painting)
    if K_g.shape != K_p.shape:
        raise ValueError(f"The kernels differ in shape: {K_g.shape} and {K_p.shape}")
    _check_genotype_weight(weight)
    K = weight * (K_g / float(K_g.sum(axis=1).mean())) + (1.0 - weight) * (
        K_p / float(K_p.sum(axis=1).mean())
    )
    return sparse.csr_matrix((K + K.T) / 2)


class GenotypePaintingPHATE(PHATE):
    """PHATE on a weighted sum of the genotype and painting affinities.

    ``fit`` takes the painting coordinates, and ``genotype`` holds the genotype
    PCA (a CSV or a DataFrame with ``sample_id``). Each view's alpha-decay
    kernel is built as PHATE builds it (graphtools, this model's ``decay`` and
    ``n_pca``): the genotype's on its top ``genotype_components`` PCs with
    ``knn``, the painting's on all its coordinates with ``painting_knn``. The
    two are mixed by :func:`mix_kernels`, ``genotype_weight`` being the
    genotype's share, and PHATE runs its diffusion, potential and MDS on the
    result as a precomputed affinity.

    There is no out-of-sample extension: ``transform`` accepts only the fit
    data. Landmarks are refused, so it is always exact.
    """

    def __init__(
        self,
        genotype: Union[pd.DataFrame, str, Path],
        genotype_weight: float,
        painting_knn: int,
        genotype_components: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.n_landmark is not None:
            raise ValueError(
                "PHATE on mixed genotype and painting affinities is exact only: set "
                f"n_landmark to none (got {self.n_landmark})"
            )
        _check_genotype_weight(genotype_weight)
        self.genotype = genotype
        self.genotype_weight = float(genotype_weight)
        self.painting_knn = int(painting_knn)
        self.genotype_components = int(genotype_components)

    def view_kernel(self, X: np.ndarray, knn: int):
        """The affinity matrix PHATE itself builds on ``X`` with ``knn``.

        PHATE's warning that this one view's graph is disconnected is silenced:
        a small ``painting_knn`` splits the painting's graph into populations by
        design, and only the mixed graph needs to be connected, which PHATE
        checks, and warns about, when it fits the mixed kernel.
        """
        view = phate.PHATE(
            knn=knn,
            decay=self.decay,
            n_pca=self.n_pca,
            n_landmark=None,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Graph is disconnected", RuntimeWarning)
            view.fit(X)
        return view.graph.kernel

    def _genotype_rows(self) -> np.ndarray:
        """The top genotype PCs, one row per fit sample, in the fit data's order."""
        geno = self.genotype
        if not isinstance(geno, pd.DataFrame):
            geno = pd.read_csv(geno, dtype={"sample_id": str})
        geno = geno.set_index(geno["sample_id"].astype(str)).drop(columns="sample_id")
        if geno.shape[1] < self.genotype_components:
            logger.warning(
                f"The genotype PCA has {geno.shape[1]} components, fewer than the "
                f"{self.genotype_components} its kernel is built on. Using all of them."
            )
        ids = [str(s) for s in self._sample_ids]
        missing = [s for s in ids if s not in geno.index]
        if missing:
            raise ValueError(
                f"{len(missing)} of the {len(ids)} painted samples have no genotype PCA row "
                f"(for example {', '.join(missing[:3])})"
            )
        return geno.loc[ids].iloc[:, : self.genotype_components].to_numpy(dtype=float)

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "GenotypePaintingPHATE":
        X_array, self._sample_ids = self._load_input_data(X)
        self._check_exact_fits_in_memory(len(X_array))
        geno = self._genotype_rows()

        logger.info(
            f"PHATE on mixed affinities: genotype weight {self.genotype_weight:g}, genotype "
            f"knn {self.knn} on {geno.shape[1]} PCs, painting knn {self.painting_knn}, "
            f"t={self.t}"
        )
        K = mix_kernels(
            self.view_kernel(geno, self.knn),
            self.view_kernel(X_array, self.painting_knn),
            self.genotype_weight,
        )
        self.model.set_params(knn_dist="precomputed_affinity", n_landmark=None)
        self.model.fit(K)
        self._fit_X = X_array
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("PHATE not fitted. Call fit() first.")
        X_array, sample_ids = self._load_input_data(X)
        if not self._is_fit_data(X_array):
            raise ValueError(
                "PHATE on mixed genotype and painting affinities cannot place new samples: "
                "it embeds only the data it was fitted on"
            )
        return self._format_output(self.model.transform(), sample_ids)

    def fit_transform(self, X, output_path=None) -> pd.DataFrame:
        self.fit(X)
        return self._format_output(self.model.transform(), self._sample_ids, output_path)
