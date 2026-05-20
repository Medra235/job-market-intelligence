# Base Python runtime for the API.
FROM python:3.12-slim-bookworm

# Send Python logs directly to Docker logs.
ENV PYTHONUNBUFFERED=1

# Install uv.
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# Compile bytecode
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1

# uv Cache
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#caching
ENV UV_LINK_MODE=copy

WORKDIR /app

# Prefer executables installed in the project virtual environment.
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
ENV PATH="/app/.venv/bin:$PATH"

# Install locked runtime dependencies first for better Docker cache reuse.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev --no-group notebooks

# Copy the application package and install the local project.
COPY src ./src
RUN uv sync --locked --no-dev --no-group notebooks

# Run the API as a non-root user.
RUN useradd --create-home --shell /usr/sbin/nologin nonroot \
    && chown -R nonroot:nonroot /app
USER nonroot

# FastAPI listens on port 8000 inside the container.
EXPOSE 8000

# Start the ASGI server.
# Uses `--host 0.0.0.0` to allow access from outside the container
CMD ["uvicorn", "job_market_intelligence.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
