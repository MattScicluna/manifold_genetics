# API reference

The public surface, for using this as a library rather than a command-line tool.

```python
from manifold_genetics import PCA, PHATE, UMAP, TSNE, DiffusionMap, Pipeline
from manifold_genetics.pipeline.runner import run_pipeline
```

Every embedding has the same three methods — `fit`, `transform`, `fit_transform`
— and each accepts a numpy array, a DataFrame, or a path to a CSV in the
standard format. That is what makes them interchangeable.

For a worked example, see the [tutorial](tutorial.ipynb).

## Running a pipeline

::: manifold_genetics.pipeline.runner.run_pipeline

::: manifold_genetics.pipeline.configfile.load_config

::: manifold_genetics.pipeline.orchestrator.Pipeline

## Results

::: manifold_genetics.pipeline.result.PipelineResult

## PCA

::: manifold_genetics.pca.flashpca.PCA

::: manifold_genetics.pca.backends.base.PCAModel

::: manifold_genetics.pca.plink.read_bed_dosages

::: manifold_genetics.pca.standardize.binom2_stats

## Embeddings

::: manifold_genetics.embeddings.base.EmbeddingBase

::: manifold_genetics.embeddings.phate.PHATE

::: manifold_genetics.embeddings.umap.UMAP

::: manifold_genetics.embeddings.tsne.TSNE

::: manifold_genetics.embeddings.diffusion_map.DiffusionMap

## Admixture

::: manifold_genetics.admixture.neural.NeuralAdmixture

::: manifold_genetics.admixture.backends.base.AdmixtureBackend

## Visualisation

::: manifold_genetics.visualization.plotting.visualize

::: manifold_genetics.visualization.plotting.plot_embedding
