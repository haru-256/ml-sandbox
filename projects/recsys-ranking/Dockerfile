# Using multi-stage build to slim down the image
FROM --platform=linux/amd64 nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04@sha256:d02c4310b6d57ca0b16cd80298bdb33a74187baafe2eccd8a6a16180ddc90802

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.6@sha256:b1e699368d24c57cda93c338a57a8c5a119009ba809305cc8e86986d4a006754 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Replicate monorepo structure so pyproject.toml's relative path deps resolve.
# pyproject.toml references ../../libs/ml_sandbox_libs, which is outside the build
# context. It is provided via an additional build context.
# vertex-job-runner is in the 'vertex' dependency group and excluded from Docker builds.
WORKDIR /workspace/projects/recsys-ranking

COPY --from=ml_sandbox_libs . /workspace/libs/ml_sandbox_libs
COPY pyproject.toml uv.lock .python-version ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra=gpu

COPY ./src ./src

# Disable runtime uv sync to prevent uv from checking missing editable dependencies at runtime.
ENV UV_NO_SYNC=1

WORKDIR /workspace/projects/recsys-ranking/src
CMD ["uv", "run", "fit.py"]
