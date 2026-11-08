"""
PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding).

Wrapper for the PHATE algorithm with a consistent API.
"""

import logging
import os
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


def _offdiag_mean(D: np.ndarray) -> float:
    n = D.shape[0]
    return float(D.sum() / (n * (n - 1))) if n > 1 else 1.0


def least_squares_scale(D_target: np.ndarray, D_init: np.ndarray) -> float:
    """The scalar ``s`` minimising ``||D_target - s * D_init||``."""
    denom = float(np.sum(D_init * D_init))
    return float(np.sum(D_target * D_init) / denom) if denom > 0 else 1.0


def metric_mds_from(
    D: np.ndarray,
    init: np.ndarray,
    *,
    solver: str = "sgd",
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Metric MDS of the distance matrix ``D`` started from ``init``.

    The same solvers ``phate.mds.embed_MDS`` uses, but that function always
    starts from classic MDS. Here the start is given, rescaled to ``D`` by a
    least-squares scalar first (the SGD solver normalizes its start anyway,
    SMACOF does not). The result is rotated onto the start by Procrustes, as
    PHATE rotates its own result onto classic MDS, so both share its scale.
    """
    from phate import mds as phate_mds
    from phate import sgd_mds
    from scipy.spatial import procrustes
    from sklearn.metrics.pairwise import euclidean_distances

    init = np.asarray(init, dtype=float)
    init = init - init.mean(axis=0)
    init = init * least_squares_scale(D, euclidean_distances(init))
    n_components = init.shape[1]
    if solver == "sgd":
        Y = sgd_mds.sgd_mds_metric(
            D, n_components=n_components, init=init, random_state=random_state
        )
    elif solver == "smacof":
        Y = phate_mds.smacof(
            D, n_components=n_components, init=init, random_state=random_state, metric=True
        )
    else:
        raise ValueError(f"Unknown MDS solver {solver!r}. Choose from: sgd, smacof")
    _, Y, _ = procrustes(init, Y)
    return Y


class InitializedPHATE(PHATE):
    """EXPERIMENTAL: PHATE whose final MDS starts from a given layout.

    PHATE's own MDS always starts from classic MDS of the potential distances.
    This subclass fits the PHATE graph and potential as usual, then runs the
    metric MDS itself from ``init_layout`` (one row per fit sample) and stores
    the result as the PHATE model's embedding, so ``transform`` and the
    landmark interpolation work unchanged. The library is not patched.

    With ``init_blend`` = alpha, the MDS target is
    ``alpha * D_init / mean + (1 - alpha) * D_potential / mean``, where
    ``D_init`` is the Euclidean distance between rows of the init layout. It
    starts from the init layout too.

    Landmarks: with ``n_landmark`` set, PHATE embeds the landmarks and places
    each sample by its transitions to them. The init layout is then averaged
    over the samples assigned to each landmark (``graph.clusters``) to give a
    landmark-level start.

    After fitting, ``init_disparity`` is the Procrustes disparity between the
    init layout and the fitted samples' embedding (0 = same shape up to
    rotation, translation and scale, 1 = unrelated).
    """

    def __init__(
        self,
        init_layout: Union[np.ndarray, pd.DataFrame],
        init_blend: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if init_blend is not None and not 0.0 <= float(init_blend) <= 1.0:
            raise ValueError(f"init_blend must be in [0, 1], got {init_blend}")
        self.init_layout = init_layout
        self.init_blend = None if init_blend is None else float(init_blend)
        self.init_disparity: Optional[float] = None

    def _init_rows(self, n_samples: int) -> np.ndarray:
        layout = self.init_layout
        if isinstance(layout, pd.DataFrame):
            if "sample_id" in layout.columns and self._sample_ids is not None:
                indexed = layout.set_index(layout["sample_id"].astype(str))
                ids = [str(s) for s in self._sample_ids]
                missing = [s for s in ids if s not in indexed.index]
                if missing:
                    raise ValueError(
                        f"The init layout has no row for {len(missing)} of the "
                        f"{len(ids)} fit samples (e.g. {missing[:3]})"
                    )
                layout = indexed.loc[ids].drop(columns="sample_id")
            else:
                layout = layout.drop(columns="sample_id", errors="ignore")
            layout = layout.to_numpy(dtype=float)
        layout = np.asarray(layout, dtype=float)
        if layout.shape != (n_samples, self.n_components):
            raise ValueError(
                f"The init layout is {layout.shape}, expected ({n_samples}, {self.n_components})"
            )
        return layout

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "InitializedPHATE":
        super().fit(X)
        self._embed_from_init()
        return self

    def fit_transform(self, X, output_path=None) -> pd.DataFrame:
        self.fit(X)
        return self._format_output(self.model.transform(), self._sample_ids, output_path)

    def _embed_from_init(self) -> None:
        import graphtools
        from scipy.spatial import procrustes
        from sklearn.metrics.pairwise import euclidean_distances

        model = self.model
        init_samples = self._init_rows(len(self._fit_X))
        # Landmark-level when PHATE uses landmarks, sample-level otherwise.
        potential = model._calculate_potential()
        if isinstance(model.graph, graphtools.graphs.LandmarkGraph):
            clusters = np.asarray(model.graph.clusters)
            init = np.vstack(
                [init_samples[clusters == c].mean(axis=0) for c in np.unique(clusters)]
            )
            logger.info(f"PHATE init: averaged the init layout over {len(init)} landmarks")
        else:
            init = init_samples

        D = euclidean_distances(potential)
        if self.init_blend is not None:
            D_init = euclidean_distances(init)
            a = self.init_blend
            D = a * D_init / _offdiag_mean(D_init) + (1 - a) * D / _offdiag_mean(D)
            logger.info(f"PHATE init: MDS on blended distances, alpha={a}")

        model.embedding = metric_mds_from(
            D, init, solver=self.mds_solver, random_state=self.random_state
        )
        _, _, self.init_disparity = procrustes(init_samples, model.transform())
        logger.info(
            f"PHATE init: Procrustes disparity from the init layout = {self.init_disparity:.4f}"
        )


KERNEL_MODES = ("kernel_sum", "kernel_product")
KERNEL_NORMS = ("degree", "markov")


def combine_kernels(
    K_a, K_b, *, mode: str = "kernel_sum", weight: float = 0.5, norm: str = "degree"
):
    """Combine two PHATE affinity matrices over the same samples into one.

    ``kernel_sum``: ``weight * N(K_a) + (1 - weight) * N(K_b)``, where ``N``
    puts both views on one scale. ``norm="degree"`` divides a kernel by its
    mean row sum, which keeps its shape, so weight 1 or 0 gives back a single
    view's PHATE exactly (PHATE row-normalizes the result). ``norm="markov"``
    row-normalizes each kernel and symmetrizes it, so every sample has the
    same total weight in both views, whatever its local density.

    ``kernel_product``: alternating diffusion (Lederman and Talmon 2018), one
    step in view a then one in view b, ``P_a @ P_b`` with ``P`` the row
    normalized kernels, symmetrized. Two samples are then close only if they
    are close in both views. ``weight`` and ``norm`` are not used.

    Returns a symmetric sparse matrix.
    """
    from scipy import sparse

    K_a, K_b = sparse.csr_matrix(K_a), sparse.csr_matrix(K_b)
    if K_a.shape != K_b.shape:
        raise ValueError(f"The kernels differ in shape: {K_a.shape} and {K_b.shape}")
    if mode not in KERNEL_MODES:
        raise ValueError(f"Unknown kernel mode {mode!r}. Choose from: {', '.join(KERNEL_MODES)}")
    if norm not in KERNEL_NORMS:
        raise ValueError(f"Unknown kernel norm {norm!r}. Choose from: {', '.join(KERNEL_NORMS)}")
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"The kernel weight must be in [0, 1], got {weight}")

    def markov(K):
        return sparse.diags(1.0 / np.asarray(K.sum(axis=1)).ravel()) @ K

    if mode == "kernel_product":
        P = markov(K_a) @ markov(K_b)
        return ((P + P.T) / 2).tocsr()

    if norm == "degree":
        N_a = K_a / float(K_a.sum(axis=1).mean())
        N_b = K_b / float(K_b.sum(axis=1).mean())
    else:
        P_a, P_b = markov(K_a), markov(K_b)
        N_a, N_b = (P_a + P_a.T) / 2, (P_b + P_b.T) / 2
    return sparse.csr_matrix(weight * N_a + (1.0 - weight) * N_b)


class CombinedKernelPHATE(PHATE):
    """EXPERIMENTAL: PHATE on a combination of two views' affinity matrices.

    One view is the input to ``fit`` (here the painting coordinates), the
    other is ``second_view`` (the genotype PCA, one row per fit sample, with
    ``sample_id``). Each view's alpha-decay kernel is built exactly as PHATE
    builds it, with this model's ``knn``, ``decay`` and ``n_pca``, the two are
    combined by :func:`combine_kernels` (``weight`` is the second view's
    share), and PHATE runs its diffusion, potential and MDS on the result as a
    precomputed affinity.

    There is no out-of-sample extension: ``transform`` accepts only the fit
    data. Landmarks are not supported, so it is exact.
    """

    def __init__(
        self,
        second_view: Union[np.ndarray, pd.DataFrame],
        mode: str = "kernel_sum",
        weight: float = 0.5,
        norm: str = "degree",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.n_landmark is not None:
            raise ValueError(
                "Combining kernels needs exact PHATE: set n_landmark to none "
                f"(got {self.n_landmark})"
            )
        self.second_view = second_view
        self.mode, self.weight, self.norm = mode, float(weight), norm
        # Fails early on a bad mode, norm or weight.
        combine_kernels(np.eye(2), np.eye(2), mode=mode, weight=self.weight, norm=norm)

    def view_kernel(self, X: np.ndarray):
        """The affinity matrix PHATE itself builds on ``X`` with these settings."""
        view = phate.PHATE(
            knn=self.knn,
            decay=self.decay,
            n_pca=self.n_pca,
            n_landmark=None,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        view.fit(X)
        return view.graph.kernel

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "CombinedKernelPHATE":
        X_array, self._sample_ids = self._load_input_data(X)
        self._check_exact_fits_in_memory(len(X_array))
        second = self._second_rows(len(X_array))

        logger.info(
            f"PHATE on combined kernels ({self.mode}, genotype weight {self.weight}, "
            f"norm {self.norm}) with knn={self.knn}"
        )
        K = combine_kernels(
            self.view_kernel(second),
            self.view_kernel(X_array),
            mode=self.mode,
            weight=self.weight,
            norm=self.norm,
        )
        self.model.set_params(knn_dist="precomputed_affinity", n_landmark=None)
        self.model.fit(K)
        self._fit_X = X_array
        self._is_fitted = True
        return self

    def _second_rows(self, n_samples: int) -> np.ndarray:
        view = self.second_view
        if isinstance(view, pd.DataFrame):
            if "sample_id" in view.columns and self._sample_ids is not None:
                indexed = view.set_index(view["sample_id"].astype(str)).drop(columns="sample_id")
                ids = [str(s) for s in self._sample_ids]
                missing = [s for s in ids if s not in indexed.index]
                if missing:
                    raise ValueError(
                        f"The second view has no row for {len(missing)} of the "
                        f"{len(ids)} fit samples (e.g. {missing[:3]})"
                    )
                view = indexed.loc[ids]
            else:
                view = view.drop(columns="sample_id", errors="ignore")
        view = np.asarray(view, dtype=float)
        if len(view) != n_samples:
            raise ValueError(f"The second view has {len(view)} rows, expected {n_samples}")
        return view

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("PHATE not fitted. Call fit() first.")
        X_array, sample_ids = self._load_input_data(X)
        if not self._is_fit_data(X_array):
            raise ValueError(
                "PHATE on combined kernels cannot place new samples: it embeds only the "
                "data it was fitted on"
            )
        return self._format_output(self.model.transform(), sample_ids)

    def fit_transform(self, X, output_path=None) -> pd.DataFrame:
        self.fit(X)
        return self._format_output(self.model.transform(), self._sample_ids, output_path)
