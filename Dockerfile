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
#
# GPU variant (for admixture.wdl): a CUDA *devel* base, because neural-admixture
# compiles a small CUDA extension with nvcc on first use, and the CUDA torch:
#   docker build --build-arg BASE=nvidia/cuda:12.1.1-devel-ubuntu22.04 \
#                --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
ARG BASE=python:3.11-slim
FROM --platform=linux/amd64 ${BASE}

# procps: Nextflow reads task metrics with `ps`.
# perl: the WRayner checker needs modules (IO::Uncompress::Gunzip) that the
# slim image's perl-base lacks; without them it dies before checking anything.
# google-cloud-cli: gsutil / bq / gcloud storage, which `acquire aou` and the
# All of Us workflows use to read the release and publish results.
# A base without Python (the CUDA one) gets the system Python, with the headers
# and compiler the CUDA extension build needs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps ca-certificates curl gnupg perl \
 && if ! command -v python > /dev/null; then \
      apt-get install -y --no-install-recommends python3 python3-pip python3-dev g++ \
      && ln -s /usr/bin/python3 /usr/local/bin/python \
      && ln -s /usr/bin/pip3 /usr/local/bin/pip; \
    fi \
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

# CPU torch keeps the image a fraction of the CUDA build's size; the GPU variant
# above passes the CUDA index.
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
