# Container for the Nextflow workflow (main.nf): the package with its admixture
# and interactive extras, and plink2 / flashpca / plink 1.9 fetched at build time,
# so a run downloads nothing.
#
#   docker build --platform linux/amd64 -t manifold-genetics:latest .
#   docker tag manifold-genetics:latest us-central1-docker.pkg.dev/<project>/<repo>/manifold-genetics:<version>
#   docker push us-central1-docker.pkg.dev/<project>/<repo>/manifold-genetics:<version>
#
# linux/amd64: flashpca is published only for Linux x86-64 (and Google Batch is amd64).
FROM --platform=linux/amd64 python:3.11-slim

# procps: Nextflow reads task metrics with `ps`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MANIFOLD_GENETICS_TOOL_DIR=/opt/manifold-tools

# CPU torch keeps the image a fraction of the CUDA build's size; for GPU
# admixture build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG EXTRAS=admixture,interactive

WORKDIR /opt/manifold-genetics
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN pip install --extra-index-url "${TORCH_INDEX}" ".[${EXTRAS}]" \
 && manifold-genetics setup \
 && chmod -R a+rX /opt/manifold-tools \
 && manifold-genetics --help > /dev/null

WORKDIR /work
