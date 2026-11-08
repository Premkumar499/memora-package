# Multi-platform production-ready Dockerfile for Memora
FROM python:3.12-slim

# Avoid prompts from debconf
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install basic OS utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project specification files first for caching
COPY pyproject.toml README.md LICENSE /app/

# Install dependencies (CPU PyTorch by default, will auto-detect CUDA if nvidia container toolkit is present)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir psutil pydantic rich typer

# Copy application source code
COPY memora /app/memora

# Install package in editable/distributable mode
RUN pip install --no-cache-dir -e .

# Expose non-root user for security best practice
RUN useradd -m -u 1000 memorauser && chown -R memorauser:memorauser /app
USER memorauser

# Default command inspects hardware and verifies readiness
ENTRYPOINT ["memora"]
CMD ["hardware"]

