# --------------------------------------------------------------
# brainrisk — slim demo image (Python only, no FreeSurfer)
# --------------------------------------------------------------
# Build:  docker build -t brainrisk-demo .
# Run:    docker run --rm brainrisk-demo make demo-preprocessing-pipeline
#
# For the full pipeline with FreeSurfer, see infrastructure/hpc_setup.md.
# --------------------------------------------------------------

# --- Stage 1: build dependencies ---
FROM python:3.12-slim AS builder

WORKDIR /app

# System deps for scipy/numpy compilation (if wheels aren't available)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE Makefile ./
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/

RUN pip install --no-cache-dir -e ".[dev]"

# --- Stage 2: slim runtime ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY pyproject.toml README.md LICENSE Makefile ./
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/

# Re-install in editable mode (registers entry points in this stage)
RUN pip install --no-cache-dir --no-deps -e .

# Default: run the demo
CMD ["make", "demo-preprocessing-pipeline"]
