# manifold-genetics

<a href="https://github.com/MattScicluna/manifold_genetics/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MattScicluna/manifold_genetics/ci.yml?branch=main&label=CI"></a>
<a href="https://coveralls.io/github/MattScicluna/manifold_genetics?branch=main"><img alt="Coverage Status" src="https://img.shields.io/coverallsCoverage/github/MattScicluna/manifold_genetics?branch=main"></a>
<a href="https://pypi.org/project/manifold-genetics/"><img alt="PyPI" src="https://img.shields.io/pypi/v/manifold-genetics"></a>
<a href="https://mattscicluna.github.io/manifold_genetics/"><img alt="Documentation" src="https://img.shields.io/badge/docs-mkdocs--material-teal"></a>

An end-to-end pipeline from PLINK genotypes to publication figures: PCA →
admixture → manifold embedding → visualisation → metrics.

<p align="center">
  <img src="assets/ukbb_phate.png" width="30%" alt="UKBB PHATE embedding coloured by self-described ancestry"/>
  <img src="assets/aou_phate.png" width="30%" alt="All of Us PHATE embedding coloured by ancestry"/>
</p>

## Installation

Requires Python 3.10–3.12. Tested on Linux and macOS.

```bash
pip install manifold-genetics
```

Admixture is an optional extra, because it pulls in torch:

```bash
pip install 'manifold-genetics[admixture]'
```

### Installation from source

```bash
pip install git+https://github.com/MattScicluna/manifold_genetics
```

## Documentation and tutorials

- **[Documentation](https://mattscicluna.github.io/manifold_genetics/)**
- **[Quickstart](https://mattscicluna.github.io/manifold_genetics/quickstart/)** —
  a full run on public data, and the input and output formats
- **[Tutorial](https://mattscicluna.github.io/manifold_genetics/tutorial/)** —
  a complete run on a simulated cohort, under a minute, nothing to download
- **[Install](https://mattscicluna.github.io/manifold_genetics/install/)** —
  optional extras, external tools, and developing on the repository

## Help

Open an [issue](https://github.com/MattScicluna/manifold_genetics/issues).

## License

BSD 3-Clause License (see [LICENSE](LICENSE)).

## Citation

If you use this package in your research, please cite:

```
[Your citation here]
```
