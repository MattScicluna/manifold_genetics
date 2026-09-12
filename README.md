# manifold-genetics

<a href="https://github.com/MattScicluna/manifold_genetics/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MattScicluna/manifold_genetics/ci.yml?branch=main&label=CI"></a>
<a href="https://coveralls.io/github/MattScicluna/manifold_genetics?branch=main"><img alt="Coverage Status" src="https://img.shields.io/coverallsCoverage/github/MattScicluna/manifold_genetics?branch=main"></a>
<a href="https://pypi.org/project/manifold-genetics/"><img alt="PyPI" src="https://img.shields.io/pypi/v/manifold-genetics"></a>
<a href="https://mattscicluna.github.io/manifold_genetics/"><img alt="Documentation" src="https://img.shields.io/badge/docs-mkdocs--material-teal"></a>

An end-to-end pipeline from PLINK genotypes to publication figures: PCA →
admixture → manifold embedding → visualisation → metrics.

**[Documentation](https://mattscicluna.github.io/manifold_genetics/)** ·
**[Quickstart](https://mattscicluna.github.io/manifold_genetics/quickstart/)** ·
**[Tutorial](https://mattscicluna.github.io/manifold_genetics/tutorial/)** (a full
run on a simulated cohort, under a minute, nothing to download)

<p align="center">
  <img src="assets/ukbb_phate.png" width="30%" alt="UKBB PHATE embedding coloured by self-described ancestry"/>
  <img src="assets/aou_phate.png" width="30%" alt="All of Us PHATE embedding coloured by ancestry"/>
</p>

## Install

```bash
pip install manifold-genetics
```

That is the whole install. PCA runs in process — it reads PLINK `.bed` directly
and computes a randomized SVD — so there is no binary to fetch and no platform it
only works on. FlashPCA remains an opt-in accelerator and writes the same
artefacts, so a model fitted by either is readable by the other.

Admixture is an optional extra, because it pulls in torch:

```bash
pip install 'manifold-genetics[admixture]'
```

## Run

```bash
manifold-genetics run config.yaml --dry-run   # print the resolved settings
manifold-genetics run config.yaml             # do the work
```

A config names two genotype sets, the labels describing them, and a colormap:

```yaml
preset: whole_cohort

data:
  fit_plink: data/fit_subset
  project_plink: data/project_subset
  labels: data/labels.csv
  colormap: colormaps/mine.json
  output_dir: outputs

pca:
  n_pcs: 50

embedding:
  method: phate
```

Which set the model is **fitted** on is the choice that defines the analysis.
`projection` fits a reference panel and places your cohort in it; `subsample`
fits and embeds a subset of your own cohort; `whole_cohort` fits a subset and
embeds everything. See
**[Configuration](https://mattscicluna.github.io/manifold_genetics/configuration/)**
for every key, and
**[Command line](https://mattscicluna.github.io/manifold_genetics/cli/)** for the
individual stages.

Input and output formats — what goes in, what comes out — are documented in the
**[Quickstart](https://mattscicluna.github.io/manifold_genetics/quickstart/)**.

## Examples

`examples/` ships a config per cohort. `examples/hgdp_1kgp/` is the one you can
run without applying for anything: download and preparation scripts included,
4,094 samples across seven genetic regions.

## Development

```bash
uv venv --python python3.11 && source .venv/bin/activate
uv sync --frozen --extra dev

uv run pytest -m "not slow and not network"   # the fast suite
uv run black src tests && uv run isort src tests && uv run flake8 src tests
```

Testing against real cohorts, the release procedure, and the safeguards this
repository uses for controlled-access data are documented under `docs/` —
`testing-real-cohorts.md`, `releasing.md` and `working-with-agents.md`.

## License

BSD 3-Clause License (see LICENSE file)

## Citation

If you use this package in your research, please cite:

```
[Your citation here]
```
