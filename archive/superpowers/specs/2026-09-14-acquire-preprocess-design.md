# `acquire` and `preprocess`: replacing `init` with two steps

Date: 2026-09-14. Supersedes the command shape in issue #123.

## The idea in one paragraph

Every cohort reaches the pipeline through two stages that today live in
`examples/`: **getting the data into one place** (`download_data.sh`,
`download_aou_data.sh`, `init hgdp`) and **filtering it** (the shared
`preprocess_cross_projection.sh` plus one `prepare_data.sh` wrapper per cohort).
The first stage is different for every cohort; the second is the same script
with different flags. A third, smaller stage — choosing which samples to fit on
— sits between them in the subsample experiments. So the package gets three
commands with that split:

- `manifold-genetics acquire <cohort>` — cohort-specific. Fetches, or is pointed
  at, the genotypes; derives the labels; writes a colormap and a config. This is
  `init` renamed and extended, and `init` goes away.
- `manifold-genetics preprocess <config> [<config>] --out DIR` — generic and
  **optional**. Reads a cohort directory, runs the shared filtering, and writes a
  new cohort directory in the same layout. Its output can be fed to `run` or
  to another `preprocess`, and skipping it entirely is a valid pipeline.
- `manifold-genetics subsample <config> --out DIR ...` — generic and optional.
  Chooses the fit samples of a cohort by label counts, a list, or geosketch.
  Same layout in and out.

The invariant that makes this composable: **a cohort directory is the unit of
exchange**; `preprocess` and `subsample` each map one to another, and `run`
consumes one. `preprocess` filters SNPs and never samples; `subsample` filters
samples and never SNPs. Neither reads a label to decide what to do with a
genotype (verified: the shell has no reference to labels, CSVs or populations,
and no `--keep`/`--remove`/`--mind`).

The old scripts under `examples/` are **kept** as the reference for what the
commands must reproduce, until each has been run once in its real environment
(UKBB on Narval, AoU in the workbench).

## The cohort directory

What `acquire` writes and `preprocess` both reads and writes. It is exactly what
`init` writes today, so `run` needs no change.

```
<dir>/
  config.yaml            # names every file below, relative to <dir>
  colormap.json          # whole_cohort   (or colormap_fit.json + colormap_project.json for projection)
  data/
    fit_subset.{bed,bim,fam}
    project_subset.{bed,bim,fam}
    labels.csv           # whole_cohort   (or fit_labels.csv + project_labels.csv for projection)
```

`config.yaml` is the entry point: it already carries `preset`, `fit_plink`,
`project_plink`, the label and colormap paths. `preprocess` takes configs, not
PLINK prefixes, so that it always knows where the labels and colormaps are.

Label rule: a label file must **cover** the `.fam` it is paired with (every
sample in the `.fam` has a row). Extra rows are allowed. This is the rule
`init custom` enforces through `_checked_labels` and that `labels_match_fam`
enforces in shell; nothing here loosens it.

## `acquire`

```
manifold-genetics acquire synthetic  [--out DIR]
manifold-genetics acquire hgdp       [--out DIR] [--archive PATH|URL] [--no-download]
manifold-genetics acquire aou        [--out DIR]
manifold-genetics acquire custom     --fit-plink P [--project-plink P] --labels CSV
                                     [--fit-labels CSV --project-labels CSV]
                                     [--preset ...] [--out DIR]
```

Each target is one function in `scaffold.py` (renamed from `init_*`), with the
same "refuse to clobber `config.yaml` unless `--force`" rule and the same
idempotence: a raw archive already on disk is not re-fetched.

| target | genotypes | labels | new work vs `init` |
|---|---|---|---|
| `synthetic` | simulated | simulated | none — rename |
| `hgdp` | Dropbox archive → `full_dataset`; plink2 `--keep` into fit (3,400 unrelated) and project (4,094 QC-passing) | `metadata.csv` → `labels.csv` + colormap | accept `--archive` that is a `gs://` URL, and recognise the workbench archive layout (see issue 1) |
| `aou` | `gsutil cp` from `gs://fc-aou-datasets-controlled/v8/microarray/plink`, FAM fix, per-chromosome split, merge → `project_subset` | BigQuery (`pandas-gbq`, needs `WORKSPACE_CDR`) → `DemographicData.tsv` → `labels.csv`; colormap from `race_ethnicity` | port of `download_aou_data.sh` (≈300 lines shell, ≈80 lines embedded Python). Untestable until workbench access returns (#124) |
| `custom` | verify the prefixes given | verify the CSVs given cover the `.fam`s; generate colormap | none — rename |

`acquire aou` keeps `aou_environment_problems()` as its first act, so outside the
workbench it fails in the first second with the list of what is missing, exactly
as `init aou` does.

UK Biobank is `acquire custom` with the paths from `mappings_private.json`.
There is no `acquire ukbb`: the package has nothing to fetch and no label
schema to derive.

## `preprocess`

```
manifold-genetics preprocess FIT_CONFIG [PROJECT_CONFIG] --out DIR
    [--preset intersect-only|harmonise]           # bundles of the flags below
    [--maf F] [--geno F] [--ld-window N] [--ld-step N] [--ld-r2 F]
    [--skip-wrayner] [--skip-giab] [--skip-hla] [--skip-ld-prune]
    [--skip-dedup] [--skip-maf] [--skip-project-maf] [--fit-has-chr-prefix]
    [--threads N] [--memory MB] [--temp-dir DIR] [--tools-dir DIR] [--cleanup]
```

One config: filter a cohort. Two configs: the first supplies `fit_plink` and its
labels, the second `project_plink` and its labels; the output config has
`preset: projection`. Both cases run the same shared script — the one-config
case passes the config's own `fit_plink` and `project_plink` as the two sides.
For a `whole_cohort` cohort those are two subsets of one dataset, so the
intersection step is a no-op and the filters apply to both. This is what makes
`preprocess` "the same script for all the code."

What it does, in order, each step skipped when its output already exists:

1. Read the config(s); resolve every path relative to its config.
2. Resolve `plink2` and `plink` through `ToolResolver`, and pass them to the
   shell explicitly (see issue 4).
3. Run `preprocess_cross_projection.sh` — shipped inside the wheel as
   `manifold_genetics/preprocessing/preprocess_cross_projection.sh` together with
   `common.sh` — with `--reference-plink` = fit, `--biobank-plink` = project,
   `--output-dir DIR/data`, and the flags. Output: `DIR/data/fit_subset.*`,
   `DIR/data/project_subset.*`.
4. Filter each input label file to the corresponding output `.fam`, write
   `DIR/data/{labels,fit_labels,project_labels}.csv`. Copy the colormap(s).
   A colormap with entries for values that were filtered out is harmless.
5. Write `DIR/config.yaml`: the input config's settings with the data paths
   pointed at `DIR/data` and `preset` set to `projection` in the two-config case.

Flag naming: the shell says `reference`/`biobank`; the CLI says `fit`/`project`
(per the naming rule: fit is the operation, project is the dataset role). So the
shell's `--skip-biobank-maf` surfaces as `--skip-project-maf`. The shell keeps
its names — it is being shipped, not rewritten.

Presets reproduce the three existing wrappers:

| preset | flags | reproduces |
|---|---|---|
| `intersect-only` | `--skip-wrayner --skip-giab --skip-hla --skip-ld-prune --skip-dedup --skip-maf` | `examples/generic/*/prepare_data.sh` |
| `harmonise` | `--skip-project-maf --cleanup` (everything on, incl. WRayner) | `examples/aou/hgdp_1kgp_proj/prepare_data.sh`, which also passes `--reference-has-chr-prefix` — a property of the data, so it stays a separate flag (`--fit-has-chr-prefix`) rather than part of the bundle |
| *(none)* | `--skip-wrayner --skip-project-maf` | `examples/ukbb/hgdp_1kgp_proj/prepare_data.sh` — spelled out, since it is neither bundle |

## `subsample`

```
manifold-genetics subsample CONFIG --out DIR
    ( --group COLUMN=PATTERN:COUNT [--group ...] [--include-rest] [--seed N]
    | --fit-samples FILE
    | --geosketch N --pca CSV )
```

Generalises `examples/aou/shared/select_samples.py` and
`examples/generic/subset/prepare_data.sh` to any cohort directory. It reads the
config's `project_plink` and its labels, chooses a fit set, writes
`DIR/data/fit_subset.*` with `plink2 --keep`, links or copies `project_subset.*`
unchanged, filters the labels, and writes a config with `preset: subsample` —
the run preset whose meaning is exactly "fit on a subset, embed the fit set"
(`embedding_input: fit`, random landmarking). The input's fit side is dropped:
given a `projection` config (HGDP fit, biobank project), the output fits on a
subset of the biobank, which is what `examples/aou/10k_WBH` does with the
intersected AoU data.

Three ways to choose, mutually exclusive:

- `--group COLUMN=PATTERN:COUNT`, repeatable. `PATTERN` is a case-insensitive
  regex matched with `str.contains` against `COLUMN` of the label file; `COUNT`
  samples are drawn from the matches with `--seed`, all of them if fewer.
  Later groups exclude samples already taken. `--include-rest` appends every
  sample no group matched. This is `select_samples.py` with the column named
  instead of guessed, which is what makes it work for UKBB
  (`--group "ethnic_background=British:10000" --group "ethnic_background=Irish:5000"`)
  and AoU (`--group "race_ethnicity=White|European:10000" ... --include-rest`)
  alike.
- `--fit-samples FILE`: a `FID IID` list, for selections made elsewhere.
  Reproduces `examples/ukbb/10k_WB_5K_Irish`.
- `--geosketch N --pca CSV`: geometric sketching on a PCA table from an earlier
  `run`. Reproduces `examples/ukbb/geosketch_phate`. Needs the `geosketch`
  extra. Listed here so the surface is settled; **implemented last**, after the
  two above have been used.

Order relative to `preprocess`: the existing flows subsample the *intersected*
data, so `preprocess` → `subsample` → `run`. Nothing prevents the other order
— they commute — but MAF filters then see the whole cohort, which is what the
published runs did.

## Equivalence with the old scripts

The stated goal is that the new commands reproduce the old scripts. Command by
command:

| old | new |
|---|---|
| `examples/hgdp_1kgp/download_data.sh` + `prepare_data.sh` | `acquire hgdp` (already claimed by `init hgdp`) |
| `examples/generic/hgdp_1kgp_proj/prepare_data.sh` | `acquire custom` ×2 → `preprocess ref/config.yaml cohort/config.yaml --preset intersect-only` |
| `examples/ukbb/hgdp_1kgp_proj/prepare_data.sh` | `acquire custom --fit-plink <gnomAD HGDP> --labels metadata.csv --out ref/`; `acquire custom --fit-plink <ukbb> --labels ukbb_labels.csv --out ukbb/`; `preprocess ref/config.yaml ukbb/config.yaml --skip-wrayner --skip-project-maf --out proj/` |
| `examples/aou/shared/download_aou_data.sh` + `hgdp_1kgp_proj/prepare_data.sh` | `acquire hgdp --archive gs://…/1KGPHGDP.tar.gz --out ref/`; `acquire aou --out aou/`; `preprocess ref/config.yaml aou/config.yaml --preset harmonise --fit-has-chr-prefix --out proj/` |
| `examples/aou/10k_WBH/prepare_data.sh` | the AoU line above, then `subsample proj/config.yaml --group "race_ethnicity=White\|European:10000" --group "race_ethnicity=Black or African American:10000" --group "race_ethnicity=Hispanic or Latino:10000" --include-rest --seed 42 --out 10k_wbh/` |
| `examples/ukbb/10k_WB_5K_Irish/prepare_data.sh` | the UKBB line above, then `subsample proj/config.yaml --fit-samples fit_samples.txt --out 10k/` |
| `examples/ukbb/geosketch_phate/prepare_data.sh` | `subsample proj/config.yaml --geosketch 50000 --pca 10k/outputs/pca.csv --out sketch/` |

The AoU case is three commands (`acquire`, `acquire`, `preprocess`), then `run`.
Decided 2026-09-14: that is the shape; no combined command.

## Issues found while specifying, and the decisions

Decisions recorded 2026-09-14. Ordered by how much they change the design.

### 1. "HGDP" is three different files — decided

| flow | reference | reachable from |
|---|---|---|
| `init hgdp`, `examples/hgdp_1kgp` | Dropbox `hgdp_1kgp_full.tar.gz` → `full_dataset`: 4,151 samples, **172,152 SNPs, already filtered and LD-pruned** | anywhere |
| UKBB | `gnomad…hgdp_tgp…LDpruned150kb_1_0.05.noHLA.unrelated` — a private prefix on Lustre | Narval only |
| AoU | `gs://fc-secure-…/Data/1KGPHGDP.tar.gz` → `extractedChrAllUnpruned`, no `chr` prefix, FIDs are `forReference<Population>` | the workbench only |

Decisions:

- **The public archive being pre-filtered is a feature.** Running `preprocess`
  on `acquire hgdp`'s output must return the same cohort — same samples, same
  variants. That is the end-to-end sanity check for the whole component and
  becomes a CI-adjacent test (see Testing). Compare on `(chr, pos, a1, a2)`
  sets and sample-ID sets, not bytes: the shell rewrites variant IDs to
  `chr:pos:ref:alt`, and `.bed` byte order is not guaranteed.

  **Observed on the real archive (Task 9, 2026-09-14, `tests/integration/test_hgdp_idempotence.py`,
  172,152 SNPs, `acquire hgdp` fit=3,400/project=4,094).** Neither parametrisation
  is exactly lossless, and both are `xfail(strict=True)` rather than passing or
  having their assertion loosened:
  - `--preset intersect-only`: 747 of 172,152 (0.43%) lost on both sides, not 0.
    Cause: the shell's `--geno 0.05` missingness filter has **no `--skip` flag** —
    it runs even under `intersect-only`. The public panel's missingness was
    computed at n=4,151; recomputed on `acquire_hgdp`'s smaller, different
    subsets (3,400 fit / 4,094 project) it newly exceeds 5% on 739 reference-side
    and 36 project-side SNPs, and the post-filter position-overlap step drops the
    union of those. Sample IDs and chromosome names (still bare `1`, not `chr1`,
    on both sides — consistent with Task 5's synthetic-cohort finding) are
    unaffected.

    **Fixed (Task 9b, 2026-09-14): `--skip-geno` / `skip_geno`.** Added to the
    shell, `flags.py` and the CLI exactly for this gap — no preset sets it, so
    existing presets' behaviour is unchanged. `--preset intersect-only
    --skip-geno` on the same archive is exactly lossless: 172,152 SNPs and the
    same samples on both sides, checked with a strict (non-`xfail`) assertion
    in `test_hgdp_idempotence.py`. The two `xfail`s above stay as they are —
    they exercise the *default* behaviour of `intersect-only` and the UKBB
    flags, which still run `--geno` (and, for the UKBB flags, GIAB/HLA/MAF/LD-
    prune) unconditionally.
  - UKBB flag set (`skip_wrayner`, `skip_project_maf`, everything else on): 70,795
    of 172,152 (41.1%) lost on both sides. Breakdown on the reference/fit side:
    GIAB+HLA region exclusion −47,714, the unconditional `--geno` filter −121,
    MAF≥0.01 on the reference −3,008 (the project side only loses its own 36 to
    `--geno`, since `--skip-biobank-maf` is set), then LD-pruning the
    already-pruned, now much-smaller intersected set at the same parameters
    (`--indep-pairwise 150 1 0.05`) removes a further 19,952 — far more than the
    "few" a re-prune of an already-pruned panel was expected to cost, because by
    that point GIAB/HLA/geno/MAF have already changed which SNPs and LD
    structure are present. HLA exclusion itself removed 0 extra (the panel is
    already `.noHLA`); GIAB accounts for essentially all of the 47,714.

  **Observed on real UKBB/HGDP data (Task 11, 2026-09-14,
  `tests/integration/test_ukbb_preprocess_idempotence.py`, `requires_private_data`).**
  Two checks, chosen instead of re-running the UKBB flag set on the already-
  intersected output (which would just repeat the lossy-LD-reprune finding
  above at 200x the cost):
  - **Test A, lossless round trip.** `--preset intersect-only --skip_geno=True`
    on the published, already-intersected cohort
    (`examples/ukbb/hgdp_1kgp_proj/data/{fit,project}_subset`, 3,340/486,748
    samples, 120,849 SNPs each) is exactly lossless: 0 samples and 0 variants
    lost on either side. Confirms Task 9b's fix generalises past HGDP: with
    every lossy step off, re-filtering an already-filtered cohort is a no-op.
  - **Test B, reproduction from raw inputs.** `acquire_custom` on the raw
    prefixes named in `mappings_private.json` (`hgdp_plink`: the gnomAD-
    derived HGDP panel, 3,340 samples x 189,783 SNPs; `ukbb_plink`: raw UKBB,
    486,748 samples x 169,829 SNPs — sample counts already equal the published
    subsets, so `acquire_custom` performs no sample-level filtering itself),
    then `preprocess` with exactly `prepare_data.sh`'s flags
    (`--skip-wrayner --skip-project-maf`, everything else on) **exactly**
    reproduces the published cohort: same sample lists and the same
    `(chr, pos, a1, a2)` sets on both sides, 3,340 x 120,849 fit and
    486,748 x 120,849 project. Per-step SNP counts: reference filtering
    189,783 → 184,681 (dedup/GIAB/HLA/geno/MAF); biobank filtering
    169,829 → 154,365 (geno only, MAF skipped); position-overlap intersection
    184,681/154,365 → 150,474 common; LD-pruning (on the reference, applied to
    both) 150,474 → 120,849 (29,625 removed). Unlike the HGDP case, this is
    not "refilter an independently-resampled subset" — the raw prefixes'
    sample sets already match the published output exactly, so it is a
    genuine from-scratch reproduction of `prepare_data.sh`'s own pipeline, not
    a second, different filtering.
  - **Memory.** The default 28,000 MB (28 GB) `--memory` OOM-killed the
    biobank-side `--set-all-var-ids`/`--make-bed` ID-standardisation step on
    486,748 samples (observed RSS 33.4 GB against an expected ~18.7 GB `.bed`)
    on a 32 GB node. `--memory 100000`, matching what
    `examples/ukbb/hgdp_1kgp_proj/prepare_data.sh` itself passes, succeeded on
    a ≥128 GB node with peak RSS 85.4 GB. `PreprocessOptions.memory` should be
    set to at least this for any real run over UKBB-sized (>400k sample)
    cohorts; the shell has no internal safeguard against under-provisioning.
  - **Resumability weakness (deferred, not fixed here).** After the OOM kill,
    relaunching the same `preprocess` call against the same `temp_dir` did
    not re-run the killed step: the shell's checkpointing
    (`if [[ -f "$OUTPUT" ]]; then skip`) only checks that an intermediate file
    *exists*, not that it is complete. The OOM had left a truncated
    `biobank_standardized_ids.bed` (7.9 GB of the expected 18.7 GB); the
    resumed run skipped straight past it, and the corruption only surfaced
    several steps later as plink's own `Error: Unexpected PLINK 1 .bed file
    size` during the SNP-intersection `--extract`. Recovery required manually
    identifying and deleting every file written at or after the truncated one
    (by timestamp) before a second resume would succeed. A future fix should
    make the shell verify each checkpoint file's expected size (or write a
    `.done` sentinel after each step) rather than trusting existence alone.
- **The AoU flow is reproduced as-is**, including the final SNP count landing
  below the shell's 100k warning. The workbench has its own HGDP+1KGP, and its
  layout differs, so `acquire hgdp --archive PATH|gs://URL` detects which
  archive it unpacked (`full_dataset.*` vs `extractedChrAllUnpruned.*`) and
  applies the `chr`-prefix fix and the `forReference` strip for the workbench
  one. The log names the layout, so nobody mistakes one panel for the other.
  If the workbench panel turns out to need filtering the public one does not,
  that is a `preprocess` flag choice on the reference side, not a new command.

### 2. Which HGDP samples are the fit set — decided: accept the difference

`acquire hgdp` fits on the 3,400 unrelated; the old AoU script fit on every
sample of `extractedChrAllUnpruned`. The unrelated subset is the defensible
choice. The AoU docs say so.

**Amendment (Task 8, 2026-09-14): the accepted difference applies to the
public archive only.** Picking the unrelated subset needs the
`filter_king_related` and QC columns of the public `metadata.csv`, and the
workbench archive carries no metadata at all -- only the population, in the
FID. So for the workbench layout `acquire hgdp` fits on every sample, exactly
as the old script did, and writes labels with `sample_id, Population` and a
generated colormap. If a relatedness list for the workbench panel ever turns
up, it is a `subsample` or `preprocess` concern, not a change to `acquire`.

### 3. Labels move before filtering — verified possible

Every wrapper today builds labels *after* intersection and filters them to the
final `.fam`. Under the split, `acquire` builds labels for the raw cohort and
`preprocess`/`subsample` re-filter them. Checked against the shell: it never
reads a label, a CSV or a population, and never subsets samples, so the
filtering is label-free by construction. The only label-dependent operation in
the old scripts is sample selection, which is now `subsample --group` and
takes its column by name. Rule preserved from `labels_match_fam`: a command
always rewrites the labels in its output directory from its input; it never
reuses a `labels.csv` it finds there because the file exists.

### 4. The shell assumes a checkout — proposed fix

Three ties to the checkout break once the script lives in `site-packages`:

- `find_plink2` / `find_plink` look under `<project_root>/bin/`. **Fix:** add
  `--plink2 PATH --plink PATH` to the shell; Python resolves both through
  `ToolResolver` and always passes them. The shell's own search stays as the
  fallback for running it by hand from a checkout.
- Two `python3 <<EOF` blocks `sys.path.insert(0, "<root>/src")` to import
  `manifold_genetics.utils`. **Fix:** add `--python PATH`; Python passes
  `sys.executable`, in which the package is already importable. Drop the
  `sys.path` lines.
- `TOOLS_DIR` (`<script>/../tools`) is where the `harmonise` preset **downloads
  at run time**: the GIAB difficult-regions bed, the WRayner checker, and the
  TOPMed reference. **Fix:** default `TOOLS_DIR` to `ToolResolver`'s cache, and
  add `manifold-genetics setup --preprocessing` to prefetch the three on a
  login node. Without that, `harmonise` needs internet where `preprocess`
  runs: true in the workbench, false on a Narval compute node. This is why the
  UKBB flow skips WRayner, and the docs say so.

Decided 2026-09-14: `setup --preprocessing` is in.

### 5. `acquire custom` fetches nothing — decided: fine

It is the bucket "get this cohort into a cohort directory, whatever that
takes"; for data you already have, that is verify and write a config. A user
with their own PLINK files can skip it and hand-write the fifteen lines.

### 6. Removing `init` — decided: remove, 0.3.0

Touches `docs/cli.md`, `docs/quickstart.md`, `docs/tutorial.ipynb` (executed on
every docs build), the README, `tests/unit/test_scaffold.py`,
`tests/integration/test_init_runs.py`, `tests/unit/test_colormap_warning.py`,
and the "Written by `manifold-genetics init …`" header in every generated
config. No alias: two names for the first command a new user runs is worse
than one rename. CHANGELOG entry.

### 7. AoU is three commands — decided

`acquire hgdp`, `acquire aou`, `preprocess`, then `run`. Each is resumable and
fails locally. A notebook cell in the AoU docs strings them together.

### 8. Sample subsetting — decided: its own command

See the `subsample` section. Three selection modes; `--group` and
`--fit-samples` first, `--geosketch` last. It is neither acquisition nor SNP
filtering, so folding it into either would break "input and output have the
same format, same script for every cohort."

### 9. `whole_cohort` through the intersection script — to verify

The one-config `preprocess` relies on the shell accepting reference and biobank
that are subsets of one dataset. The 50k common-SNP abort is fine (a cohort
shares all its SNPs with itself); MAF is computed per side, a mild
inconsistency for two subsets of one cohort, not an error. The synthetic
round-trip test settles it.

## Testing

- **Unit:** the flag-assembly function (`config(s) + preset + overrides →
  argv`) is pure and gets a table test per preset, asserting the exact argv
  each old wrapper passed. The config-rewrite step likewise.
- **Integration, fast:** `acquire synthetic` → `preprocess` (one config,
  `--preset intersect-only`) → `subsample --group` → `run --dry-run`, on the
  simulated cohort, with the real shell and real plink2. Proves the
  cohort-directory round trip and issue 9. `preprocess` on two synthetic
  configs proves the projection case.
- **Integration, `network`:** the idempotence check from issue 1 —
  `acquire hgdp` → `preprocess --preset intersect-only` yields the same
  samples and the same `(chr, pos, a1, a2)` set. The one test that exercises
  the real filter on real data without private access.
- **Integration, `requires_private_data`:** the UKBB reproduction from the
  equivalence table, checked against the existing `examples/ukbb/hgdp_1kgp_proj/data`
  by SNP count and sample count. This is the only real-biobank test available.
- **Not tested:** `acquire aou` beyond `aou_environment_problems()`, until #124.

## Out of scope

Labels for cohorts that are not HGDP, AoU or synthetic (the user supplies a
CSV); a combined AoU command (issue 7); rewriting the shell in Python (decided
in #123); deleting the `examples/` scripts (kept as the reference until each
flow has been reproduced where it runs).

## Order of work

1. `preprocess` and the shell changes of issue 4, with the synthetic
   round-trip test. Testable now, on Narval.
2. `acquire` as a rename of `init`, plus `--archive gs://` and layout
   detection. The HGDP idempotence test.
3. `subsample` with `--group` and `--fit-samples`; UKBB reproduction
   (`requires_private_data`).
4. `acquire aou` — the port of `download_aou_data.sh`. Written against the
   old script, run when workbench access returns (#124).
5. `--geosketch`, docs pages (Preprocessing, and the AoU notebook cell),
   `init` removal, CHANGELOG, 0.3.0.
