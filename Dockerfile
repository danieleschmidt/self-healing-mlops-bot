# Multi-stage Dockerfile for production deployment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash botuser

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black isort flake8 mypy

# Copy source code
COPY . .

# Set ownership
RUN chown -R botuser:botuser /app

USER botuser

# Default command for development
CMD ["python", "-m", "self_healing_bot.web.app"]

# Production stage
FROM base as production

# Copy only necessary files
COPY self_healing_bot/ ./self_healing_bot/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for logs and keys
RUN mkdir -p /app/logs /app/keys && \
    chown -R botuser:botuser /app

USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command
CMD ["python", "-m", "self_healing_bot.web.app"]

# Multi-architecture build stage
FROM production as multi-arch

# Install additional tools for different architectures
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        echo "Optimizing for ARM64"; \
    elif [ "$(uname -m)" = "x86_64" ]; then \
        echo "Optimizing for AMD64"; \
    fi

# Default target is production
FROM production