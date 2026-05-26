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
    df.insert(0, "sample_id", sample_ids)
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
    df.insert(0, "sample_id", sample_ids)
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
    return pd.DataFrame(
        {
            "sample_id": [f"SAMPLE_{i:03d}" for i in range(n_samples)],
            "Population": np.random.choice(["PopA", "PopB", "PopC"], n_samples),
            "Region": np.random.choice(["North", "South"], n_samples),
        }
    )


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
        "Population": {"PopA": "#FF0000", "PopB": "#00FF00", "PopC": "#0000FF"},
        "Region": {"North": "#FF6B6B", "South": "#4ECDC4"},
    }


@pytest.fixture
def colormap_json(temp_dir, colormap_data):
    """Colormap JSON file."""
    colormap_file = temp_dir / "colormap.json"
    with open(colormap_file, "w") as f:
        json.dump(colormap_data, f)
    return colormap_file


@pytest.fixture
def numeric_pca_data():
    """PCA dataset with integer (numeric) sample_ids, mimicking FlashPCA IID behavior."""
    np.random.seed(42)
    n_samples, n_pcs = 50, 10
    df = pd.DataFrame(
        np.random.randn(n_samples, n_pcs),
        columns=[f"dim_{i+1}" for i in range(n_pcs)],
    )
    df.insert(0, "sample_id", np.arange(n_samples, dtype=np.int64))
    return df


@pytest.fixture
def numeric_pca_csv(temp_dir, numeric_pca_data):
    """PCA CSV file with integer sample_ids."""
    pca_file = temp_dir / "numeric_pca.csv"
    numeric_pca_data.to_csv(pca_file, index=False)
    return pca_file


@pytest.fixture
def numeric_labels_data():
    """Labels for 50 samples using integer sample_ids (stored as strings in CSV)."""
    np.random.seed(42)
    n_samples = 50
    return pd.DataFrame(
        {
            "sample_id": [str(i) for i in range(n_samples)],
            "Population": np.random.choice(["PopA", "PopB", "PopC"], n_samples),
            "Region": np.random.choice(["North", "South"], n_samples),
        }
    )


@pytest.fixture
def numeric_labels_csv(temp_dir, numeric_labels_data):
    """Labels CSV file with string-form integer sample_ids."""
    labels_file = temp_dir / "numeric_labels.csv"
    numeric_labels_data.to_csv(labels_file, index=False)
    return labels_file


def _create_plink_files(temp_dir: Path, prefix_name: str):
    """Helper function to create minimal dummy PLINK files.

    Args:
        temp_dir: Directory to create files in
        prefix_name: Name for the PLINK file prefix (e.g., 'dummy', 'fit', 'transform')

    Returns:
        String path to PLINK prefix
    """
    import struct

    n_samples = 50
    n_snps = 100
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]

    prefix = temp_dir / prefix_name

    # Create .fam file (sample information)
    with open(f"{prefix}.fam", "w") as f:
        for sample_id in sample_ids:
            # FID IID father mother sex phenotype
            f.write(f"{sample_id} {sample_id} 0 0 0 -9\n")

    # Create .bim file (variant information)
    with open(f"{prefix}.bim", "w") as f:
        for i in range(n_snps):
            # chr snp_id genetic_pos physical_pos allele1 allele2
            f.write(f"1 SNP_{i:04d} 0 {i+1} A T\n")

    # Create .bed file (genotype data in binary format)
    # PLINK .bed format: magic bytes (0x6c, 0x1b, 0x01) followed by genotype data
    with open(f"{prefix}.bed", "wb") as f:
        # Write magic bytes (SNP-major mode)
        f.write(struct.pack("BBB", 0x6C, 0x1B, 0x01))

        # Write genotype data with some random variation
        # Each SNP needs (n_samples+3)//4 bytes
        # PLINK .bed encoding: 00=hom ref, 01=missing, 10=het, 11=hom alt
        bytes_per_snp = (n_samples + 3) // 4

        # Generate random genotypes with variety
        for snp_idx in range(n_snps):
            # Create random genotypes (mix of 00, 10, 11 to avoid NaN in PCA)
            genotype_bytes = []
            for byte_idx in range(bytes_per_snp):
                # Each byte encodes 4 samples (2 bits per sample)
                # Generate random byte with mix of genotypes
                random_byte = (snp_idx * 17 + byte_idx * 37) % 256  # Deterministic pseudo-random
                # Avoid 01 (missing) by masking: convert 01 to 00
                for bit_pair in range(4):
                    shift = bit_pair * 2
                    geno = (random_byte >> shift) & 0b11
                    if geno == 0b01:  # Missing - convert to hom ref
                        random_byte &= ~(0b11 << shift)  # Clear bits
                genotype_bytes.append(random_byte)
            f.write(bytes(genotype_bytes))

    return str(prefix)


@pytest.fixture
def dummy_plink_files(temp_dir):
    """Create minimal dummy PLINK files for backend tests.

    These files have valid format but minimal data (50 samples, 100 SNPs).
    They're sufficient for backends that just need sample IDs.
    """
    return _create_plink_files(temp_dir, "dummy")


@pytest.fixture
def fit_plink_files(temp_dir):
    """Create PLINK files named 'fit' for precomputed admixture backend tests."""
    return _create_plink_files(temp_dir, "fit")


@pytest.fixture
def transform_plink_files(temp_dir):
    """Create PLINK files named 'transform' for precomputed admixture backend tests."""
    return _create_plink_files(temp_dir, "transform")
