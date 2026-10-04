# Container for the workflows (main.nf, the WDL workflows): the package with its
# aou, admixture and interactive extras, plink2 / flashpca / plink 1.9 and the GIAB
# difficult-regions bed fetched at build time, and the Google Cloud CLI. A run
# downloads nothing, so it works on networks that reach only Google services.
#
#   docker build --platform linux/amd64 -t manifold-genetics:latest .
#   docker tag manifold-genetics:latest us-central1-docker.pkg.dev/<project>/<repo>/manifold-genetics:<version>
#   docker push us-central1-docker.pkg.dev/<project>/<repo>/manifold-genetics:<version>
#
# linux/amd64: flashpca is published only for Linux x86-64 (and Google Batch is amd64).
FROM --platform=linux/amd64 python:3.11-slim

# procps: Nextflow reads task metrics with `ps`.
# google-cloud-cli: gsutil / bq / gcloud storage, which `acquire aou` and the
# All of Us workflows use to read the release and publish results.
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps ca-certificates curl gnupg \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends google-cloud-cli \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MANIFOLD_GENETICS_TOOL_DIR=/opt/manifold-tools

# CPU torch keeps the image a fraction of the CUDA build's size; for GPU
# admixture build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG EXTRAS=aou,admixture,interactive

WORKDIR /opt/manifold-genetics
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN pip install --extra-index-url "${TORCH_INDEX}" ".[${EXTRAS}]" \
 && manifold-genetics setup \
 && python -c "from manifold_genetics.preprocessing.references import GIAB_URL, _install_giab, default_tools_dir; from manifold_genetics.utils.tools import fetch_url; t = default_tools_dir() / 'giab/GRCh38_alldifficultregions.bed'; t.parent.mkdir(parents=True, exist_ok=True); _install_giab(GIAB_URL, t, fetch_url)" \
 && python -c "from manifold_genetics.preprocessing.references import default_tools_dir, ensure_wrayner; ensure_wrayner(default_tools_dir())" \
 && chmod -R a+rX /opt/manifold-tools \
 && manifold-genetics --help > /dev/null

WORKDIR /work
