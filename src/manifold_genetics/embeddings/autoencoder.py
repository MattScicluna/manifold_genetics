"""
Autoencoder: a plain (non-variational) MLP autoencoder with a low-dimensional bottleneck.

A neural baseline with no neighbourhood graph and no KL term, trained with mean
squared reconstruction error on the same input as the other methods (the PCA
coordinates). Encoder and decoder mirror each other: ``hidden`` layer widths
with ELU activations; the bottleneck is linear. Training uses Adam (learning
rate 1e-3) on minibatches of 128. Inputs are centred and divided by one global
standard deviation, which keeps the relative variance of the PCs (standardising
each PC separately would inflate the noisy trailing PCs to the same weight as
PC1). Needs PyTorch (``pip install 'manifold-genetics[autoencoder]'``).
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import EmbeddingBase, optional_import

logger = logging.getLogger(__name__)

_BATCH_SIZE = 128
_LEARNING_RATE = 1e-3


def _torch():
    return optional_import("torch", "pip install 'manifold-genetics[autoencoder]'")


class Autoencoder(EmbeddingBase):
    """MLP autoencoder embedding.

    Args:
        n_components: Bottleneck size.
        hidden: Hidden layer widths of the encoder (the decoder mirrors them).
        epochs: Training epochs.
        random_state: Seed.
    """

    def __init__(
        self,
        n_components: int = 2,
        hidden: Sequence[int] = (256, 128, 64),
        epochs: int = 300,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.hidden = tuple(int(h) for h in hidden)
        self.epochs = int(epochs)
        self.encoder = None

    @staticmethod
    def _mlp(sizes):
        nn = _torch().nn
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(nn.ELU())
        return nn.Sequential(*layers)

    def _scaled(self, X_array):
        return _torch().from_numpy((np.asarray(X_array, dtype=np.float32) - self._mu) / self._sd)

    def fit(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> "Autoencoder":
        torch = _torch()
        X_array, self._sample_ids = self._load_input_data(X)
        X_array = np.asarray(X_array, dtype=np.float32)
        self._mu = X_array.mean(0)
        self._sd = float((X_array - self._mu).std()) + 1e-8
        Xs = self._scaled(X_array)

        seed = 0 if self.random_state is None else int(self.random_state)
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        d = Xs.shape[1]
        self.encoder = self._mlp([d, *self.hidden, self.n_components])
        decoder = self._mlp([self.n_components, *reversed(self.hidden), d])
        opt = torch.optim.Adam(
            [*self.encoder.parameters(), *decoder.parameters()], lr=_LEARNING_RATE
        )

        logger.info(
            f"Training autoencoder {d}->{self.hidden}->{self.n_components} for {self.epochs} epochs"
        )
        n = Xs.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for start in range(0, n, _BATCH_SIZE):
                xb = Xs[perm[start : start + _BATCH_SIZE]]
                loss = torch.nn.functional.mse_loss(decoder(self.encoder(xb)), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._is_fitted = True
        return self

    def _encode(self, X, output_path=None) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Autoencoder must be fitted before transform")
        X_array, sample_ids = self._load_input_data(X)
        with _torch().no_grad():
            Y = self.encoder(self._scaled(X_array)).numpy()
        return self._format_output(Y, sample_ids, output_path)

    def transform(self, X: Union[np.ndarray, pd.DataFrame, str, Path]) -> pd.DataFrame:
        return self._encode(X)

    def fit_transform(self, X, output_path=None) -> pd.DataFrame:
        return self.fit(X)._encode(X, output_path)
