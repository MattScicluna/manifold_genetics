# Python API

Most people should use the [command line](cli.md) — it is what the pipeline is
built around. This page is for driving it from a notebook or another package.

```python
from manifold_genetics import run_pipeline, load_config, PCA, PHATE
```

## What is public

`manifold_genetics.__all__` is the contract. Everything in it is importable from
the top level and will not change without a deprecation:

| name | what it is |
|---|---|
| `run_pipeline` | run a whole pipeline from keyword arguments |
| `load_config` | read a config file into those arguments |
| `PCA` | fit and project principal components |
| `PHATE`, `UMAP`, `TSNE`, `DiffusionMap` | embeddings |
| `NeuralAdmixture` | ancestry proportions |
| `visualize`, `plot_embedding` | figures |
| `Pipeline` | the orchestrator, if you want the stages without the runner |

**Everything else is an implementation detail** and may change in a patch
release — the PCA backends, the PLINK reader, the standardisation helpers, the
admixture backend ABC. They exist to serve the surface above, not to be called
directly. If you find yourself reaching for one, that is worth reporting as a
missing piece of the public API rather than working around.

The package ships `py.typed`, so these signatures are visible to type checkers.

!!! note "The version number is doing real work here"

    At 0.2.x the API is not frozen. The surface above is deliberately small so
    that it *can* be kept stable while the rest moves.

## Embeddings are interchangeable

Every embedding has the same three methods — `fit`, `transform`, `fit_transform`
— and each accepts a numpy array, a DataFrame, or a path to a CSV in the
standard `sample_id, dim_1 … dim_n` format. That is what makes them swappable,
and what lets you start from principal components you already have.

For a worked example, see the [tutorial](tutorial.ipynb).

## Running a pipeline

::: manifold_genetics.pipeline.runner.run_pipeline

::: manifold_genetics.pipeline.configfile.load_config

::: manifold_genetics.pipeline.orchestrator.Pipeline

::: manifold_genetics.pipeline.result.PipelineResult

## PCA

::: manifold_genetics.pca.flashpca.PCA

## Embeddings

::: manifold_genetics.embeddings.base.EmbeddingBase

::: manifold_genetics.embeddings.phate.PHATE

::: manifold_genetics.embeddings.umap.UMAP

::: manifold_genetics.embeddings.tsne.TSNE

::: manifold_genetics.embeddings.diffusion_map.DiffusionMap

## Admixture

::: manifold_genetics.admixture.neural.NeuralAdmixture

## Visualisation

::: manifold_genetics.visualization.plotting.visualize

::: manifold_genetics.visualization.plotting.plot_embedding
