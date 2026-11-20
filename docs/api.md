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

### Optional baselines

These wrap other groups' published implementations and need extra installs:

| Method | `--method` | Install |
|---|---|---|
| DREAMS (Kury, Kobak & Damrich, TMLR 2026) | `dreams` | `pip install "openTSNE @ git+https://github.com/berenslab/DREAMS.git@tp"` |
| FloDR (Baraka & Probst, arXiv:2607.26278) | `flodr` | `pip install 'manifold-genetics[flodr]'` (Python 3.11 or later) |
| Autoencoder (plain MLP, no KL term) | `autoencoder` | `pip install 'manifold-genetics[autoencoder]'` |
| popVAE (Battey, Coffing & Kern, G3 2021) | `popvae` | A Python 3.10 / TensorFlow 2.15 environment with popVAE (<https://github.com/kr-colab/popvae>) and `bed-reader` installed; give its interpreter as `$POPVAE_PYTHON` and `popvae.py` as `$POPVAE_SCRIPT` |

popVAE embeds genotypes rather than PCA coordinates (pass `--genotypes PLINK_PREFIX`), and, like t-SNE, it cannot project new samples.

::: manifold_genetics.embeddings.dreams.DREAMS

::: manifold_genetics.embeddings.flodr.FloDR

::: manifold_genetics.embeddings.autoencoder.Autoencoder

::: manifold_genetics.embeddings.popvae.PopVAE

::: manifold_genetics.embeddings.base.EmbeddingBase

## Admixture

::: manifold_genetics.admixture.neural.NeuralAdmixture

## Metrics

The geographic and admixture metrics compare distances between sampled pairs of samples in the embedding with distances in another space, using a Spearman correlation.

::: manifold_genetics.metrics.geographic.compute_geographic_preservation

::: manifold_genetics.metrics.admixture.compute_admixture_preservation

k-NN preservation measures local structure instead: the share of each sample's nearest neighbours in the PCA input that stay among its nearest neighbours in the embedding.

::: manifold_genetics.metrics.knn.compute_knn_preservation

## Visualisation

::: manifold_genetics.visualization.plotting.visualize

::: manifold_genetics.visualization.plotting.plot_embedding

::: manifold_genetics.visualization.plotting.plot_embedding_3d

::: manifold_genetics.visualization.plotting.visualize_3d

::: manifold_genetics.visualization.plotting.plot_admixture_embedding_3d
