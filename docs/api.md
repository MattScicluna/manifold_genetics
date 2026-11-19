# Python API

```python
from manifold_genetics import (
    run_pipeline, load_config, Pipeline, PCA, NeuralAdmixture,
    PHATE, UMAP, TSNE, DiffusionMap, visualize, plot_embedding,
)
```

## Pipeline

::: manifold_genetics.pipeline.runner.run_pipeline

::: manifold_genetics.pipeline.configfile.load_config

::: manifold_genetics.pipeline.orchestrator.Pipeline

::: manifold_genetics.pipeline.result.PipelineResult

## PCA

::: manifold_genetics.pca.flashpca.PCA

## Embeddings

::: manifold_genetics.embeddings.phate.PHATE

::: manifold_genetics.embeddings.umap.UMAP

::: manifold_genetics.embeddings.tsne.TSNE

::: manifold_genetics.embeddings.diffusion_map.DiffusionMap

::: manifold_genetics.embeddings.base.EmbeddingBase

## Admixture

::: manifold_genetics.admixture.neural.NeuralAdmixture

## Metrics

The geographic and admixture metrics compare distances between sampled pairs of samples in the embedding with distances in another space, using a Spearman correlation.

::: manifold_genetics.metrics.geographic.compute_geographic_preservation

::: manifold_genetics.metrics.admixture.compute_admixture_preservation

Rank correlations do not change when an embedding squeezes each group into a tight cluster, because the order of distances survives the squeeze. Separation inflation is scale-aware: it asks whether groups look more separated (detached clusters) or less separated (merged) in the embedding than in a reference space such as the PCA input or admixture proportions.

::: manifold_genetics.metrics.separation.compute_separation_inflation

## Visualisation

::: manifold_genetics.visualization.plotting.visualize

::: manifold_genetics.visualization.plotting.plot_embedding

::: manifold_genetics.visualization.plotting.plot_embedding_3d

::: manifold_genetics.visualization.plotting.visualize_3d

::: manifold_genetics.visualization.plotting.plot_admixture_embedding_3d
