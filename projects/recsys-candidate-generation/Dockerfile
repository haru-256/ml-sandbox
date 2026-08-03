# Using multi-stage build to slim down the image
FROM --platform=linux/amd64 nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04@sha256:070f8f2672df1b05b84c0409a5fd1d54ddfd646e5b9d8dee7878131271b563fc

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.6@sha256:b1e699368d24c57cda93c338a57a8c5a119009ba809305cc8e86986d4a006754 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Replicate monorepo structure so pyproject.toml's relative path deps resolve.
# pyproject.toml references ../../libs/ml_sandbox_libs, which is outside the build
# context. It is provided via an additional build context (see compose.yaml
# additional_contexts / Makefile docker-build --build-context).
# Note: vertex-job-runner is in the 'vertex' dependency group and excluded from
# Docker builds, so it does not need to be copied.
WORKDIR /workspace/projects/recsys-candidate-generation

# Copy local dependency source (needed by uv sync for editable install)
COPY --from=ml_sandbox_libs . /workspace/libs/ml_sandbox_libs

# Copy project metadata for dependency resolution
COPY pyproject.toml uv.lock .python-version ./

# Install Python dependencies (project itself is installed at runtime by `uv run`)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra=gpu

# Copy project source
COPY ./src ./src

# Disable runtime uv sync to prevent uv from checking missing editable dependencies at runtime.
ENV UV_NO_SYNC=1

WORKDIR /workspace/projects/recsys-candidate-generation/src
CMD ["uv", "run", "fit.py"]
