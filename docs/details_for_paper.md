# HGDP+1KGP Data Processing Pipeline

This document provides a comprehensive description of the data processing pipeline used to prepare the HGDP+1000 Genomes Project (1KGP) dataset for manifold genetics analysis. This documentation is intended for publication and reproducibility purposes.

## Table of Contents

1. [Data Source](#1-data-source)
2. [Quality Control Filters](#2-quality-control-filters)
3. [LD Pruning](#3-ld-pruning)
4. [Region Exclusions](#4-region-exclusions)
5. [Sample Subsetting](#5-sample-subsetting)
6. [Final Dataset Properties](#6-final-dataset-properties)
7. [Label Files](#7-label-files)
8. [File Formats](#8-file-formats)
9. [Reproducibility](#9-reproducibility)

---

## 1. Data Source

### gnomAD v3.1.2 HGDP+1KGP Joint Call Set

The dataset is derived from the Genome Aggregation Database (gnomAD) version 3.1.2, which includes a joint variant call set of the Human Genome Diversity Project (HGDP) and the 1000 Genomes Project (1KGP).

**Source:** [gnomAD v3.1.2](https://gnomad.broadinstitute.org/)

**Original Sample Composition:**
- Total samples: 4,151
  - HGDP samples: ~929
  - 1KGP samples: ~3,165
- Genome build: GRCh38 / hg38
- Initial variant count: Millions (before filtering)

**Citation:**
> Chen et al. (2020). "A genome-wide mutational constraint map quantified from variation in 76,156 human genomes." *Nature*, 581(7809), 434-443.

---

## 2. Quality Control Filters

The following quality control (QC) filters were applied to ensure high-quality genetic variants:

### 2.1 Variant Quality Filters

1. **PASS Filters Only**
   - Only variants that passed all gnomAD quality filters
   - Removes low-quality and ambiguous variant calls

2. **SNPs Only**
   - Single nucleotide polymorphisms (SNPs) only
   - Excludes insertions, deletions, and complex variants

3. **No Duplicate Positions**
   - Removed variants at duplicate genomic positions
   - Ensures unique variant identifiers

4. **Missingness Threshold**
   - Maximum missing data per variant: ≤5%
   - Ensures robust genotype calls across samples

5. **Minor Allele Frequency (MAF)**
   - Minimum MAF: ≥0.01 (1%)
   - Removes very rare variants to focus on common polymorphisms

### 2.2 Example Commands

```bash
# Using bcftools for VCF filtering (example)
bcftools view \
  --include 'FILTER="PASS" && TYPE="snp"' \
  --min-af 0.01:minor \
  input.vcf.gz \
  -O z -o filtered.vcf.gz

# Using PLINK2 for missingness filtering
plink2 \
  --vcf filtered.vcf.gz \
  --geno 0.05 \
  --maf 0.01 \
  --snps-only \
  --make-bed \
  --out qc_filtered
```

---

## 3. LD Pruning

Linkage disequilibrium (LD) pruning was performed to obtain a set of approximately independent genetic variants.

### 3.1 Parameters

- **Window size:** 500 kb
- **Step size:** 50 variants (sliding window)
- **r² threshold:** 0.05
- **Method:** Variance inflation factor (VIF) based pruning

### 3.2 Rationale

LD pruning removes redundant genetic information due to correlation between nearby variants. This ensures:
- Independent variant assumption for PCA and admixture analysis
- Computational efficiency
- Reduced noise from highly correlated variants

### 3.3 Commands

```bash
# Step 1: Identify independent variants
plink2 \
  --bfile qc_filtered \
  --indep-pairwise 500 50 0.05 \
  --out ld_pruned

# Output: ld_pruned.prune.in (variants to keep)
#         ld_pruned.prune.out (variants to remove)

# Step 2: Extract independent variants
plink2 \
  --bfile qc_filtered \
  --extract ld_pruned.prune.in \
  --make-bed \
  --out pruned_dataset
```

### 3.4 Result

- **Final SNP count:** 172,152 independent SNPs
- **Average spacing:** ~17 kb between variants (given ~3 Gb genome)

---

## 4. Region Exclusions

Specific genomic regions were excluded to avoid confounding population structure analyses:

### 4.1 Low Complexity Regions

Low complexity regions (LCRs) are genomic segments with repetitive or simple sequence composition that may harbor alignment artifacts or sequencing errors.

**Source:** UCSC Genome Browser low complexity tracks
- Simple repeats
- Microsatellites
- Homopolymers

### 4.2 HLA Region

The Human Leukocyte Antigen (HLA) region on chromosome 6 exhibits extreme polymorphism and strong selection, which can dominate population structure signals.

**Excluded region:**
- Chromosome: 6
- Start: 28,477,797 bp
- End: 33,448,354 bp
- Genome build: GRCh38

**Rationale:**
- HLA diversity reflects pathogen-driven selection rather than demographic history
- Removal prevents HLA from dominating PCA and admixture analyses

### 4.3 Example Commands

```bash
# Create exclusion BED file
cat > exclude_regions.bed << EOF
chr6	28477797	33448354	HLA
EOF

# Exclude regions using PLINK2
plink2 \
  --bfile pruned_dataset \
  --exclude range exclude_regions.bed \
  --make-bed \
  --out final_dataset
```

---

## 5. Sample Subsetting

Two sample subsets were created for different analysis purposes:

### 5.1 Fit Subset (Training Set)

**Purpose:** Model training (PCA, admixture fitting)

**Selection criteria:**
- Unrelated individuals only
- Filter: `Related == 'Unrelated'` from gnomAD metadata

**Sample count:** 3,400 samples

**Rationale:**
- Removes cryptic relatedness
- Ensures independence assumption for population genetics models
- Provides clean population structure signal

### 5.2 Transform Subset (Test Set)

**Purpose:** Model application (project new samples into learned space)

**Selection criteria:**
- All QC-passing samples
- Filters applied:
  - `filter_pca_outlier == False` (not a PCA outlier)
  - `hard_filtered == False` (passed hard QC filters)
  - `filter_contaminated == False` (not contaminated)

**Sample count:** 4,094 samples

**Rationale:**
- Includes related individuals and population outliers
- Tests model generalization
- Represents real-world sample diversity

### 5.3 Population Composition

Both subsets contain samples from 7 major genetic regions:

1. **Africa** (AFR)
2. **Americas** (AMR)
3. **Central/South Asia** (CSA)
4. **East Asia** (EAS)
5. **Europe** (EUR)
6. **Middle East** (MID)
7. **Oceania** (OCN)

### 5.4 Commands

```bash
# Create fit indices (unrelated samples)
python3 << EOF
import pandas as pd

# Read gnomAD metadata
metadata = pd.read_csv('gnomad_derived_metadata_with_filtered_sampleids.csv')

# Filter for unrelated samples
unrelated = metadata[metadata['Related'] == 'Unrelated']

# Write FID IID format for PLINK
unrelated[['sample_id', 'sample_id']].to_csv(
    'fit_indices.txt',
    sep='\t',
    header=False,
    index=False
)
EOF

# Create fit subset with PLINK2
plink2 \
  --bfile final_dataset \
  --keep fit_indices.txt \
  --make-bed \
  --out fit_subset

# Create transform indices (all QC-passing samples)
python3 << EOF
import pandas as pd

metadata = pd.read_csv('gnomad_derived_metadata_with_filtered_sampleids.csv')

# Filter for QC-passing samples
qc_pass = metadata[
    (metadata['filter_pca_outlier'] == False) &
    (metadata['hard_filtered'] == False) &
    (metadata['filter_contaminated'] == False)
]

qc_pass[['sample_id', 'sample_id']].to_csv(
    'transform_indices.txt',
    sep='\t',
    header=False,
    index=False
)
EOF

# Create transform subset with PLINK2
plink2 \
  --bfile final_dataset \
  --keep transform_indices.txt \
  --make-bed \
  --out transform_subset
```

---

## 6. Final Dataset Properties

### 6.1 Sample Counts

| Subset | Sample Count | Description |
|--------|--------------|-------------|
| Fit | 3,400 | Unrelated samples for model training |
| Transform | 4,094 | All QC-passing samples for model application |

### 6.2 Variant Properties

- **Total SNPs:** 172,152
- **Average MAF:** >0.01
- **LD status:** Pruned (r² < 0.05 within 500 kb)
- **Genome coverage:** Genome-wide (excluding HLA and LCRs)

### 6.3 Genetic Ancestry Distribution

The samples span 7 continental genetic regions with representation from 54 distinct populations (specific population counts vary between fit and transform subsets).

### 6.4 File Sizes

- **Fit subset:** ~70 MB (PLINK binary format)
- **Transform subset:** ~85 MB (PLINK binary format)
- **Metadata:** ~1 MB (CSV)
- **Colormap:** ~2 KB (JSON)
- **Total package:** ~183 MB

---

## 7. Label Files

Label files provide population and genetic region annotations for each sample.

### 7.1 Label File Structure

**Format:** CSV (comma-separated values)

**Columns:**
1. `sample_id`: Unique sample identifier (matches PLINK FID/IID)
2. `Population`: Fine-grained population label (e.g., "Yoruba", "Han", "French")
3. `Genetic_region_merged`: Broad genetic region (AFR, AMR, CSA, EAS, EUR, MID, OCN)

**Example:**
```csv
sample_id,Population,Genetic_region_merged
HG00096,British,EUR
HG00097,British,EUR
NA18501,Yoruba,AFR
NA18502,Yoruba,AFR
```

### 7.2 Label Extraction

Labels are extracted from gnomAD metadata and filtered to match the sample subsets:

```python
import pandas as pd

# Read metadata
metadata = pd.read_csv('gnomad_derived_metadata_with_filtered_sampleids.csv')

# Read fit indices
fit_indices = pd.read_csv('fit_indices.txt', sep='\t', header=None, names=['FID', 'IID'])
fit_samples = set(fit_indices['IID'])

# Extract labels for fit subset
fit_labels = metadata[metadata['sample_id'].isin(fit_samples)][
    ['sample_id', 'Population', 'Genetic_region_merged']
]

fit_labels.to_csv('hgdp_fit_labels.csv', index=False)
```

### 7.3 Colormap

A JSON colormap provides consistent color coding for visualizations:

**Format:** JSON dictionary mapping genetic regions to hex colors

**Example:**
```json
{
  "AFR": "#E69F00",
  "AMR": "#56B4E9",
  "CSA": "#009E73",
  "EAS": "#F0E442",
  "EUR": "#0072B2",
  "MID": "#D55E00",
  "OCN": "#CC79A7"
}
```

---

## 8. File Formats

### 8.1 PLINK Binary Format

Genotype data is stored in PLINK binary format (`.bed`, `.bim`, `.fam`):

**`.bed` (Binary genotype file):**
- Stores genotypes in compact binary format
- 2 bits per genotype (00=homozygous ref, 01=missing, 10=heterozygous, 11=homozygous alt)

**`.bim` (Variant information file):**
- Tab-delimited text file
- Columns: Chromosome, Variant ID, Genetic distance (0), Position, Allele 1, Allele 2

**`.fam` (Sample information file):**
- Tab-delimited text file
- Columns: Family ID, Individual ID, Paternal ID, Maternal ID, Sex, Phenotype

**Reference:** [PLINK 1.9 File Format](https://www.cog-genomics.org/plink/1.9/formats)

### 8.2 CSV Format

Label files use standard CSV format with header row:
- Delimiter: comma (`,`)
- Encoding: UTF-8
- Line endings: Unix (`\n`)

### 8.3 JSON Format

Colormap uses standard JSON format:
- Encoding: UTF-8
- Indentation: 2 spaces (human-readable)

---

## 9. Reproducibility

### 9.1 Software Versions

The following software was used in data processing:

| Tool | Version | Purpose |
|------|---------|---------|
| bcftools | 1.17+ | VCF filtering |
| PLINK2 | 2.00a3.7+ | Genotype processing, LD pruning |
| Python | 3.8+ | Sample subsetting, label extraction |
| pandas | 1.3+ | Metadata manipulation |

### 9.2 Complete Processing Pipeline

A complete processing script is available at:
- `scripts/prepare_data.sh` (processes downloaded data)
- `scripts/package_data_for_dropbox.sh` (packages raw data)

### 9.3 Data Availability

The processed dataset is distributed as a compressed archive:
- **File:** `hgdp_1kgp_full.tar.gz`
- **Size:** ~183 MB
- **MD5 checksum:** (computed during packaging)

**Download:** [Dropbox link provided in README.md]

### 9.4 Processing Time

Estimated processing time on a single CPU core:
- LD pruning: ~10-15 minutes
- Sample subsetting: <1 minute
- Total: ~15-20 minutes

### 9.5 Hardware Requirements

Minimum requirements:
- RAM: 16 GB
- Storage: 500 MB
- CPU: Single core sufficient

---

## References

1. **gnomAD v3.1.2:**
   - Chen et al. (2020). "A genome-wide mutational constraint map quantified from variation in 76,156 human genomes." *Nature*, 581(7809), 434-443.
   - https://gnomad.broadinstitute.org/

2. **HGDP:**
   - Bergström et al. (2020). "Insights into human genetic variation and population history from 929 diverse genomes." *Science*, 367(6484), eaay5012.

3. **1000 Genomes Project:**
   - 1000 Genomes Project Consortium (2015). "A global reference for human genetic variation." *Nature*, 526(7571), 68-74.

4. **PLINK2:**
   - Chang et al. (2015). "Second-generation PLINK: rising to the challenge of larger and richer datasets." *GigaScience*, 4(1), 7.

---

## Contact

For questions about this data processing pipeline, please open an issue on the GitHub repository or contact the maintainers.

**Last updated:** December 2025
