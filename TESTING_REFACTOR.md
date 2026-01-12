# Testing Stage 2 Refactor

This document explains how to test the refactored example scripts to verify they work correctly without running expensive admixture computation.

## Why Test?

The Stage 2 refactor consolidated all example scripts to use the shared pipeline runner. We need to verify that:
1. All scripts run without errors
2. PCA completes successfully
3. Embeddings complete successfully
4. Output files are created in expected locations

We skip admixture computation because it's expensive (hours of compute time).

## Test Commands

### HGDP+1KGP (Mila cluster - automated)

This test is running automatically and will be documented in REFACTOR_PROGRESS.md

```bash
cd /path/to/manifold_genetics
source .venv/bin/activate
bash examples/hgdp_1kgp/run_pipeline.sh --skip-admixture
```

Expected outputs:
- `examples/hgdp_1kgp/outputs/pca/fit_pca_50.csv`
- `examples/hgdp_1kgp/outputs/pca/transform_pca_50.csv`
- `examples/hgdp_1kgp/outputs/embeddings/phate_2d.csv`
- `examples/hgdp_1kgp/outputs/figures/pca/*.png`
- `examples/hgdp_1kgp/outputs/figures/embeddings/*.png`

### UKBB 10k_WB_5K_Irish (YOUR SERVER - manual)

**Run this on your UKBB server to verify the refactor:**

```bash
# Navigate to manifold_genetics directory
cd /path/to/manifold_genetics
source .venv/bin/activate

# Run UKBB pipeline without expensive admixture
bash examples/ukbb/10k_WB_5K_Irish/run_pipeline.sh --skip-admixture --skip-metrics

# Verify outputs exist
ls examples/ukbb/10k_WB_5K_Irish/outputs/pca/fit_pca_20.csv
ls examples/ukbb/10k_WB_5K_Irish/outputs/pca/transform_pca_20.csv
ls examples/ukbb/10k_WB_5K_Irish/outputs/embeddings/phate_2d.csv
ls examples/ukbb/10k_WB_5K_Irish/outputs/figures/pca/*.png
ls examples/ukbb/10k_WB_5K_Irish/outputs/figures/embeddings/*.png
```

**What to check:**
1. ✓ Script completes without errors
2. ✓ PCA step runs (should be fast if cached)
3. ✓ PHATE embedding step completes
4. ✓ Visualization figures are created
5. ✓ No admixture files are generated (we're skipping it)

**Expected runtime:** ~2-5 minutes (mostly PCA and embedding)

### UKBB hgdp_1kgp_proj (Optional - if you want to test cross-projection)

```bash
cd /path/to/manifold_genetics
source .venv/bin/activate
bash examples/ukbb/hgdp_1kgp_proj/run_pipeline.sh --skip-admixture --skip-metrics
```

## What Changed in Stage 2

All wrapper scripts now pass through additional arguments using `"$@"`:

```bash
# Before (Stage 1):
bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
    --mode projection \
    --fit-plink "$FIT_PLINK" \
    # ... other args

# After (Stage 2):
bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
    --mode projection \
    --fit-plink "$FIT_PLINK" \
    # ... other args
    "$@"  # ← Pass through any extra args like --skip-admixture
```

This allows testing with `--skip-admixture`, `--skip-metrics`, etc.

## If Tests Pass

If HGDP and UKBB tests both pass, we can:
1. Mark Stage 2 as validated ✅
2. Proceed to Stage 3: Admixture Backend Interface
3. Use backend interface to properly test with precomputed admixture

## If Tests Fail

If a test fails:
1. Check the error message
2. Verify all required data files exist
3. Check the wrapper script passes arguments correctly
4. Report the error for investigation

## Next Steps

After both HGDP and UKBB tests pass:
- Proceed to Stage 3: Create admixture backend interface
- This will allow us to test with precomputed admixture instead of skipping it
- Final Stage 4: Comprehensive test suite with reproducibility checks
