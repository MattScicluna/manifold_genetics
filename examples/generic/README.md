# Generic Biobank Analysis Pipeline

This directory contains generic pipeline templates for analyzing biobank data with the manifold-genetics package. The UKBB pipelines (`examples/ukbb/`) are specific instances of these templates.

## Prerequisites

- Your data in PLINK format (`.bed`, `.bim`, `.fam`)
- Pre-created label CSV files with `sample_id` column and metadata columns
- Population colormap in JSON format
- Virtual environment activated (`source .venv/bin/activate`)

**Important**: The generic pipeline scripts do NOT create labels from metadata. You must create label CSV files yourself before running the pipeline. The UKBB examples show how to do this in wrapper scripts.

## Two Workflow Types

### 1. Cross-Projection (`hgdp_1kgp_proj/`)

**Use when:** You want to project your biobank samples into a reference space (e.g., HGDP+1KGP as reference).

**What it does:**
- Intersects SNPs between reference and biobank datasets
- Aligns alleles and creates standardized subsets
- Trains PCA/admixture models on reference data
- Projects biobank samples into the reference-trained space
- Generates separate visualizations for reference and biobank populations

**Steps:**

1. **Prepare PLINK data:** Intersects SNPs, aligns alleles, creates subsets

   ```bash
   cd examples/generic/hgdp_1kgp_proj

   # Create intersected PLINK files
   bash prepare_data.sh \
       --reference-plink /path/to/hgdp/data \
       --biobank-plink /path/to/biobank/data \
       --output-dir ./data \
       --memory 100000 \
       --threads 4
   ```

2. **Create label files:** Extract labels for intersected samples

   ```bash
   # You must create these yourself from your metadata
   # fit_labels.csv: Labels for reference (HGDP) samples
   # project_labels.csv: Labels for biobank samples
   # See examples/ukbb/hgdp_1kgp_proj/prepare_data.sh for an example
   ```

3. **Run pipeline:** PCA, admixture, embedding with reference-trained models

   ```bash
   bash run_pipeline.sh \
       --fit-plink ./data/fit_subset \
       --project-plink ./data/project_subset \
       --fit-labels ./data/fit_labels.csv \
       --project-labels ./data/project_labels.csv \
       --fit-colormap /path/to/fit_colormap.json \
       --project-colormap /path/to/project_colormap.json \
       --output-dir ./outputs \
       --n-pcs 20 \
       --k-min 2 --k-max 10 \
       --embedding phate --knn 100 --t 3 \
       --admixture-group-column Population \
       --threads 4 \
       --skip-metrics
   ```

**Example:** See `examples/ukbb/hgdp_1kgp_proj/` for UKBB projected into HGDP+1KGP space.

---

### 2. Internal Subset (`subset/`)

**Use when:** You want to split your own biobank data into fit/project subsets for internal validation.

**What it does:**
- Extracts fit and project subsets from a single dataset
- No SNP intersection needed (same dataset, same SNPs)
- Trains models on fit subset, projects remaining samples
- Useful for validation or stratified analysis within your biobank

**Steps:**

1. **Create sample lists:** FID IID format (tab-separated, no header)

   ```
   FID001  IID001
   FID002  IID002
   ...
   ```

2. **Prepare data:** Extracts subsets, creates labels

   ```bash
   cd examples/generic/subset

   bash prepare_data.sh \
       --plink /path/to/biobank/data \
       --fit-samples ./fit_samples.txt \
       --project-samples ./project_samples.txt \ # if we dont pass, assumes we are projecting on all samples.
       --metadata /path/to/metadata.csv \
       --output-dir ./data \
       --memory 100000 \
       --threads 4
   ```

3. **Run pipeline:** PCA, admixture, embedding with fit-trained models

   ```bash
   bash run_pipeline.sh \
       --fit-plink ./data/fit_subset \
       --project-plink ./data/project_subset \
       --fit-labels ./data/fit_labels.csv \ # you have to create this seperately
       --project-labels ./data/project_labels.csv \ # you have to create this seperately
       --colormap /path/to/colormap.json \
       --output-dir ./outputs \
       --n-pcs 20 \
       --k-min 2 --k-max 10 \
       --embedding phate --knn 100 \
       --admixture-group-column Population \
       --threads 4
   ```

**Example:** See `examples/ukbb/10k_WB_5K_Irish/` for UKBB 10K White British + 5K Irish subset analysis.

---

## Input Format Requirements

### PLINK Files

Standard binary PLINK format:
- `.bed` - genotype data (binary)
- `.bim` - variant information (6 columns: chr, rsid, cm, bp, A1, A2)
- `.fam` - sample information (6 columns: FID, IID, father, mother, sex, phenotype)

**Example:** `examples/hgdp_1kgp/data/hgdp_fit.{bed,bim,fam}`

### Labels CSV

Must include `sample_id` column matching PLINK IID. Additional columns for population/region/ancestry labels.

```csv
sample_id,Population,Region
HGDP00001,Yoruba,Africa
HGDP00002,Han,EastAsia
SAMPLE001,British,Europe
```

**Example:** `examples/hgdp_1kgp/data/hgdp_project_labels.csv`

### Colormap JSON

Maps label values to hex colors for visualization.

```json
{
  "Population": {
    "Yoruba": "#238B45",
    "Han": "#3182BD",
    "British": "#F46D43"
  },
  "Region": {
    "Africa": "#238B45",
    "EastAsia": "#3182BD",
    "Europe": "#F46D43"
  }
}
```

**Example:** `examples/colormaps/hgdp_1kgp.json`

---

## Creating Wrapper Scripts (Recommended Pattern)

The generic scripts handle PLINK processing only. For your biobank, create wrapper scripts that:
1. Read data paths from a config file (`mappings.json`)
2. Call the generic script to create PLINK subsets
3. Filter label files to match the PLINK subsets

### Configuration File Structure

**Purpose:** Single source of truth for all data paths

**Location:** `examples/your_biobank/workflow_name/data/mappings.json`

**Format:** JSON with absolute or relative paths (relative resolved from PROJECT_ROOT)

#### Cross-Projection Config

For projecting your biobank into a reference space (e.g., HGDP+1KGP):

```json
{
  "reference_plink": "/path/to/hgdp/data",
  "biobank_plink": "/path/to/biobank/data",
  "fit_labels": "examples/hgdp_1kgp/data/hgdp_fit_labels.csv",
  "project_labels": "examples/your_biobank/data/biobank_labels.csv"
}
```

**Keys:**
- `reference_plink`: Reference dataset PLINK prefix (full dataset)
- `biobank_plink`: Your biobank PLINK prefix (full dataset)
- `fit_labels`: Full reference labels CSV (all samples, with sample_id column)
- `project_labels`: Full biobank labels CSV (all samples, with sample_id column)

**Note:** Pass FULL label files. The wrapper will filter them to match samples after SNP intersection.

#### Internal Subset Config

For splitting your biobank into fit/project subsets:

```json
{
  "biobank_plink": "/path/to/biobank/data",
  "labels": "examples/your_biobank/data/all_labels.csv",
  "fit_samples": "examples/your_biobank/subset/data/fit_samples.txt"
}
```

**Keys:**
- `biobank_plink`: Your biobank PLINK prefix (full dataset)
- `labels`: Full biobank labels CSV (all samples, with sample_id column)
- `fit_samples`: Sample list for training set (FID IID format, tab-separated, no header)

**Note:** Only ONE labels file. The wrapper will create fit_labels.csv and project_labels.csv by filtering.

### Why This Pattern?

1. **Single config file:** All paths in `mappings.json`
2. **Full labels as input:** No pre-filtering needed, just pass complete label files
3. **Wrapper handles filtering:** Filters labels to match PLINK subsets after processing
4. **Flexible paths:** Supports absolute and relative paths
5. **Clear separation:** Generic scripts = PLINK only, wrappers = labels
6. **Easy adaptation:** Users only update mappings.json for their environment

### See UKBB Examples

Study `examples/ukbb/` for complete working implementations:
- `examples/ukbb/hgdp_1kgp_proj/` - Cross-projection wrapper
- `examples/ukbb/10k_WB_5K_Irish/` - Internal subset wrapper

---

## Command-Line Arguments

### prepare_data.sh (Cross-Projection)

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--reference-plink` | Yes | Reference dataset PLINK prefix | - |
| `--biobank-plink` | Yes | Biobank dataset PLINK prefix | - |
| `--reference-metadata` | Yes | Reference labels CSV | - |
| `--biobank-metadata` | Yes | Biobank metadata (CSV/TSV) | - |
| `--output-dir` | No | Output directory | `./data` |
| `--temp-dir` | No | Temporary directory | `./data/temp` |
| `--memory` | No | plink2 memory limit (MB) | `100000` |
| `--threads` | No | Number of threads | `4` |

### prepare_data.sh (Subset)

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--plink` | Yes | Input PLINK prefix | - |
| `--fit-samples` | Yes | Fit samples list (FID IID format) | - |
| `--project-samples` | No | Project samples list (uses all if not provided) | - |
| `--metadata` | Yes | Metadata file (CSV/TSV) | - |
| `--output-dir` | No | Output directory | `./data` |
| `--fit-labels-out` | No | Output fit labels CSV | `./data/fit_labels.csv` |
| `--project-labels-out` | No | Output project labels CSV | `./data/project_labels.csv` |
| `--memory` | No | plink2 memory limit (MB) | `100000` |
| `--threads` | No | Number of threads | `4` |

### run_pipeline.sh (Both Workflows)

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--fit-plink` | Yes | Fit PLINK prefix | - |
| `--project-plink` | Yes | Project PLINK prefix | - |
| `--fit-labels` | Yes | Fit labels CSV | - |
| `--project-labels` | Yes | Project labels CSV | - |
| `--fit-colormap` | Yes* | Fit colormap JSON | - |
| `--project-colormap` | Yes* | Project colormap JSON | - |
| `--colormap` | Yes* | Single colormap (sets both fit and project) | - |
| `--output-dir` | No | Output directory | `./outputs` |
| `--n-pcs` | No | Number of PCA components | `20` |
| `--k-min` | No | Min admixture K | `2` |
| `--k-max` | No | Max admixture K | `10` |
| `--embedding` | No | Embedding method (phate, umap, tsne, dm) | `phate` |
| `--knn` | No | KNN parameter | `100` |
| `--t` | No | Diffusion time (for PHATE) | `3` |
| `--n-landmark` | No | Number of landmarks (for PHATE) | - |
| `--random-landmarking` | No | Use random landmarking | `false` |
| `--embedding-input` | No | Embedding input (fit or all) | `all` |
| `--admixture-group-column` | No | Metadata column for admixture grouping | - |
| `--threads` | No | Number of threads | `4` |
| `--neuraladmixture-batch-size` | No | Neural admixture batch size | `400` |
| `--embed-batch-size` | No | Embedding batch size | - |
| `--num-gpus` | No | Number of GPUs for acceleration | - |
| `--skip-metrics` | No | Skip metrics calculation | `false` |

\* Either `--colormap` OR both `--fit-colormap` and `--project-colormap` are required.

---

## Output Structure

```
outputs/
├── pca/
│   ├── fit_pca_20.csv              # PCA coordinates (fit samples × 20 dims)
│   └── transform_pca_20.csv        # PCA coordinates (project samples × 20 dims)
├── admixture/
│   ├── fit.2.csv                   # Ancestry proportions K=2 (fit samples)
│   ├── transform.2.csv             # Ancestry proportions K=2 (project samples)
│   ├── fit.3.csv                   # K=3
│   └── ...                         # K=4 to K=10
├── embeddings/
│   └── phate_2d.csv                # 2D embedding coordinates (all samples × 2)
├── figures/
│   ├── pca/                        # PCA scatter plots
│   ├── admixture/                  # Admixture bar plots
│   └── embeddings/                 # Embedding scatter plots
└── metrics/
    ├── geographic.json             # Geographic distance preservation (if applicable)
    └── admixture.json              # Admixture proportion preservation
```

---

## Memory Considerations

For large datasets (>100K samples or >1M SNPs):

1. **Increase memory limit:**
   ```bash
   --memory 150000  # 150 GB for plink2 operations
   ```

2. **Use landmarking for PHATE:**
   ```bash
   --n-landmark 10000 --random-landmarking
   ```

3. **Reduce PCA components:**
   ```bash
   --n-pcs 10  # Use fewer components
   ```

4. **Process in stages:** Run prepare_data.sh first, verify intermediate files, then run pipeline.

**Example (AoU):** See `examples/aou/hgdp_1kgp_proj/` for handling 400K+ samples.

---

## UKBB as Example

The UKBB pipelines (`examples/ukbb/`) are thin wrappers around these generic scripts. They:
1. Read UKBB-specific paths from `data/mappings.json`
2. Call the generic scripts with appropriate arguments

**Study the UKBB implementation** to understand how to adapt these templates for your biobank.

### UKBB Structure

```
examples/ukbb/
├── hgdp_1kgp_proj/           # Cross-projection: UKBB → HGDP+1KGP space
│   ├── data/
│   │   └── mappings.json     # UKBB-specific paths (not tracked)
│   ├── prepare_data.sh       # Wrapper → calls generic hgdp_1kgp_proj/prepare_data.sh
│   └── run_pipeline.sh       # Wrapper → calls generic hgdp_1kgp_proj/run_pipeline.sh
└── 10k_WB_5K_Irish/          # Internal subset: 10K British + 5K Irish
    ├── data/
    │   ├── fit_samples.txt   # Pre-selected sample IDs
    │   └── mappings.json     # UKBB-specific paths (not tracked)
    ├── prepare_data.sh       # Wrapper → calls generic subset/prepare_data.sh
    └── run_pipeline.sh       # Wrapper → calls generic subset/run_pipeline.sh
```

---

## Troubleshooting

### "plink2 not found"
Run `setup.sh` from project root to download tools:
```bash
cd /path/to/manifold_genetics
bash setup.sh
```

### "Virtual environment not activated"
Activate before running pipelines:
```bash
source .venv/bin/activate
```

### "Missing required files"
Ensure prepare_data.sh completed successfully before running run_pipeline.sh.

### "Out of memory"
Increase `--memory` flag or reduce dataset size. Use landmarking for large embeddings.

### "Allele mismatch errors" (cross-projection)
The prepare_data.sh script automatically handles allele flipping. If errors persist, check for multi-allelic SNPs or encoding issues in your PLINK files.

---

## Further Reading

- **HGDP+1KGP example:** `examples/hgdp_1kgp/` - Public reference dataset
- **AoU example:** `examples/aou/` - Large-scale biobank (400K+ samples)
- **UKBB example:** `examples/ukbb/` - Wrapper implementation pattern
- **Colormaps:** `examples/colormaps/` - Example color schemes for visualization
- **Documentation:** `docs/` - Detailed implementation and API documentation

---

## Citation

If you use this pipeline in your research, please cite:
```
[Citation to be added upon publication]
```
