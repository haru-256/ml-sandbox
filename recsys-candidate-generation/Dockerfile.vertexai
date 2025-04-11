# Using multi-stage build to slim down the image
FROM --platform=linux/amd64 nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-install-project --no-dev --extra=gpu
COPY ./src /src
WORKDIR /src

CMD ["uv", "run", "fit.py"]
