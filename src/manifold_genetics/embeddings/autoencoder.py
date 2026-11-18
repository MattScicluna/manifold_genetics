"""
Autoencoder: a plain (non-variational) MLP autoencoder with a low-dimensional bottleneck.

A neural baseline with no neighbourhood graph and no KL term, trained with mean
squared reconstruction error on the same input as the other methods (the PCA
coordinates). Encoder and decoder mirror each other: ``hidden`` layer widths
with ELU activations; the bottleneck is linear. Inputs are standardised per
column before training. Needs PyTorch (``pip install 'manifold-genetics[autoencoder]'``).
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import EmbeddingBase

logger = logging.getLogger(__name__)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The autoencoder needs PyTorch: `pip install 'manifold-genetics[autoencoder]'`."
        ) from exc
    return torch


class Autoencoder(EmbeddingBase):
    """MLP autoencoder embedding.

    Args:
        n_components: Bottleneck size.
        hidden: Hidden layer widths of the encoder (decoder is reversed).
        epochs: Training epochs.
        batch_size: Minibatch size.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        random_state: Seed (torch and numpy).
    """

    def __init__(
        self,
        n_components: int = 2,
        hidden: Sequence[int] = (256, 128, 64),
        epochs: int = 300,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.hidden = tuple(int(h) for h in hidden)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.encoder = None

    def _mlp(self, sizes):
        torch = _import_torch()
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(torch.nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(torch.nn.ELU())
        return torch.nn.Sequential(*layers)

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "Autoencoder":
        torch = _import_torch()
        X_array, self._sample_ids = self._load_input_data(X)
        X_array = np.asarray(X_array, dtype=np.float32)
        self._mu = X_array.mean(0)
        self._sd = X_array.std(0) + 1e-8
        Xs = torch.from_numpy((X_array - self._mu) / self._sd)

        seed = 0 if self.random_state is None else int(self.random_state)
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        d = Xs.shape[1]
        self.encoder = self._mlp([d, *self.hidden, self.n_components])
        decoder = self._mlp([self.n_components, *reversed(self.hidden), d])
        params = list(self.encoder.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        logger.info(f"Training autoencoder {d}->{self.hidden}->{self.n_components} for {self.epochs} epochs")
        n = Xs.shape[0]
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for start in range(0, n, self.batch_size):
                xb = Xs[perm[start : start + self.batch_size]]
                loss = torch.nn.functional.mse_loss(decoder(self.encoder(xb)), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            if (epoch + 1) % 50 == 0:
                logger.info(f"  epoch {epoch + 1}: reconstruction MSE {total / n:.4f}")
        self._final_loss = total / n
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        torch = _import_torch()
        if not self._is_fitted:
            raise RuntimeError("Autoencoder must be fitted before transform")
        X_array, sample_ids = self._load_input_data(X)
        Xs = torch.from_numpy(((np.asarray(X_array, dtype=np.float32) - self._mu) / self._sd))
        with torch.no_grad():
            Y = self.encoder(Xs).numpy()
        return self._format_output(Y, sample_ids)

    def fit_transform(self, X, output_path=None) -> pd.DataFrame:
        self.fit(X)
        out = self.transform(X)
        if output_path:
            from ..utils.io import write_embedding_csv

            write_embedding_csv(out, output_path)
        return out
