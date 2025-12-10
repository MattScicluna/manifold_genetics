# Tests for manifold-genetics

Unit tests for the manifold-genetics package using pytest.

## Test Coverage

We test **downstream analysis steps** with small dummy data:
- ✅ **I/O utilities** - Reading/writing CSVs, labels, colormaps
- ✅ **Embeddings** - PHATE, UMAP, t-SNE, Diffusion Maps
- ✅ **Visualization** - Plotting and figure generation

**Not tested** (require expensive external tools):
- ❌ PCA (requires flashPCA)
- ❌ Admixture (requires neural-admixture training)

## Running Tests

### Install test dependencies
```bash
source .venv/bin/activate
pip install pytest pytest-cov
```

### Run all tests
```bash
pytest tests/
```

### Run with coverage
```bash
pytest tests/ --cov=manifold_genetics --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_io.py
pytest tests/test_embeddings.py
pytest tests/test_visualization.py
```

### Run specific test
```bash
pytest tests/test_embeddings.py::TestPHATE::test_fit_transform
```

### Verbose output
```bash
pytest tests/ -v
```

## Test Data

Tests use small synthetic datasets generated in `conftest.py`:
- **50 samples × 10 PCs** - Fast PCA-like data
- **50 samples × 2 dims** - Small embeddings
- **3 populations, 2 regions** - Test labels
- **Colormap JSON** - Test color schemes

All test data is generated on-the-fly, no fixtures needed.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures
├── test_io.py               # I/O utilities tests
├── test_embeddings.py       # PHATE, UMAP, t-SNE, DM tests
└── test_visualization.py    # Plotting tests
```

## Writing New Tests

1. Add fixtures to `conftest.py` if needed
2. Create test classes for organization
3. Use descriptive test names: `test_<what>_<scenario>`
4. Keep tests fast (< 1 second each)
5. Use small data (50-100 samples)

Example:
```python
class TestNewFeature:
    def test_basic_functionality(self, temp_dir):
        """Test basic use case."""
        # Your test here
        assert result == expected
```
