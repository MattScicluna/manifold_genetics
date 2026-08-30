# Cross-cohort test fixtures

50-sample fit/project label + colormap + geographic files for the parametrized
pipeline contract test (`test_pipeline_output_layout`, PR 5). Sample IDs
(`SAMPLE_000`…`SAMPLE_049`) match `tests/fixtures/admixture/`.

- `fit_labels.csv` — `sample_id`, `Population` (HGDP-style)
- `project_labels.csv` — `sample_id`, `self_described_ancestry` (UKBB-style)
- `fit_colormap.json` / `project_colormap.json` — keyed on the above columns
- `geographic.csv` — `sample_id`, `latitude`, `longitude`

Regenerate: `python tests/fixtures/cross_cohort/generate.py`
