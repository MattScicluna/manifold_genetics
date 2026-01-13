# Precomputed Admixture Fixtures

This directory contains precomputed admixture proportions for testing.

## Files

- `fit.2.csv` - Admixture proportions for K=2 (fit subset, 50 samples)
- `fit.3.csv` - Admixture proportions for K=3 (fit subset, 50 samples)
- `transform.2.csv` - Admixture proportions for K=2 (transform subset, 50 samples)
- `transform.3.csv` - Admixture proportions for K=3 (transform subset, 50 samples)

## Format

Each file is a CSV with:
- Column 1: `sample_id` (SAMPLE_000 through SAMPLE_049)
- Columns 2+: `component_1`, `component_2`, ..., `component_K`

All component values sum to 1.0 per row.

## Usage in Tests

These fixtures allow integration tests to bypass slow neural-admixture computation:

```python
import pytest
from manifold_genetics.admixture import NeuralAdmixture
from manifold_genetics.admixture.backends import PrecomputedAdmixtureBackend

@pytest.mark.integration
def test_pipeline_with_precomputed_admixture(tmp_path):
    # Use precomputed backend instead of real neural-admixture
    backend = PrecomputedAdmixtureBackend(
        fixture_dir="tests/fixtures/admixture"
    )
    admix = NeuralAdmixture(k_min=2, k_max=3, backend=backend)

    # Run pipeline...
```

## Regeneration

To regenerate these fixtures from real data:

```python
import pandas as pd
from manifold_genetics.admixture import NeuralAdmixture

# Run real admixture (slow!)
admix = NeuralAdmixture(k_min=2, k_max=3)
admix.fit("path/to/fit.plink", output_dir="temp/")
fit_q = admix.transform("path/to/fit.plink", output_prefix="temp/fit")
transform_q = admix.transform("path/to/transform.plink", output_prefix="temp/transform")

# Copy to fixtures directory
# fit_q[2] -> tests/fixtures/admixture/fit.2.csv
# fit_q[3] -> tests/fixtures/admixture/fit.3.csv
# etc.
```
