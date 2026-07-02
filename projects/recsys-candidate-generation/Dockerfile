# Using multi-stage build to slim down the image
FROM --platform=linux/amd64 nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04@sha256:d02c4310b6d57ca0b16cd80298bdb33a74187baafe2eccd8a6a16180ddc90802

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.14@sha256:3362a526af7eca2fcd8604e6a07e873fb6e4286d8837cb753503558ce1213664 /uv /uvx /bin/
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

WORKDIR /workspace/projects/recsys-candidate-generation/src
CMD ["uv", "run", "fit.py"]
