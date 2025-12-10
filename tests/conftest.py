"""
Pytest fixtures for manifold-genetics tests.

We test downstream steps (embeddings, visualization) with small dummy data.
PCA and Admixture are not tested as they require expensive external tools.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil
import json


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def small_pca_data():
    """Small PCA dataset (50 samples × 10 PCs) for fast tests."""
    np.random.seed(42)
    n_samples = 50
    n_pcs = 10
    data = np.random.randn(n_samples, n_pcs) * 0.1
    cols = [f"dim_{i+1}" for i in range(n_pcs)]
    df = pd.DataFrame(data, columns=cols)
    # Add sample_id column as first column
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    df.insert(0, 'sample_id', sample_ids)
    return df


@pytest.fixture
def small_pca_csv(temp_dir, small_pca_data):
    """Small PCA CSV file for testing."""
    pca_file = temp_dir / "pca_10.csv"
    small_pca_data.to_csv(pca_file, index=False)
    return pca_file


@pytest.fixture
def small_embedding_data():
    """Small 2D embedding (50 samples × 2 dims) for visualization tests."""
    np.random.seed(42)
    n_samples = 50
    embedding = np.random.randn(n_samples, 2)
    df = pd.DataFrame(embedding, columns=["dim_1", "dim_2"])
    # Add sample_id column as first column
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    df.insert(0, 'sample_id', sample_ids)
    return df


@pytest.fixture
def small_embedding_csv(temp_dir, small_embedding_data):
    """Small embedding CSV file."""
    embedding_file = temp_dir / "embedding.csv"
    small_embedding_data.to_csv(embedding_file, index=False)
    return embedding_file


@pytest.fixture
def labels_data():
    """Labels for 50 samples with 2 categorical variables."""
    np.random.seed(42)
    n_samples = 50
    return pd.DataFrame({
        "sample_id": [f"SAMPLE_{i:03d}" for i in range(n_samples)],
        "Population": np.random.choice(["PopA", "PopB", "PopC"], n_samples),
        "Region": np.random.choice(["North", "South"], n_samples),
    })


@pytest.fixture
def labels_csv(temp_dir, labels_data):
    """Labels CSV file."""
    labels_file = temp_dir / "labels.csv"
    labels_data.to_csv(labels_file, index=False)
    return labels_file


@pytest.fixture
def colormap_data():
    """Colormap for test labels."""
    return {
        "Population": {
            "PopA": "#FF0000",
            "PopB": "#00FF00",
            "PopC": "#0000FF"
        },
        "Region": {
            "North": "#FF6B6B",
            "South": "#4ECDC4"
        }
    }


@pytest.fixture
def colormap_json(temp_dir, colormap_data):
    """Colormap JSON file."""
    colormap_file = temp_dir / "colormap.json"
    with open(colormap_file, 'w') as f:
        json.dump(colormap_data, f)
    return colormap_file
