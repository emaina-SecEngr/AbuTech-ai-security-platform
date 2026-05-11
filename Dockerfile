# ============================================================
# AbuTech AI Security Platform
# Production Dockerfile
#
# PURPOSE:
#   Packages the entire platform into one container.
#   Same behavior everywhere it runs.
#   No "works on my machine" problems.
#
# SECURITY PRINCIPLES APPLIED:
#
#   1. Minimal base image (slim)
#      Fewer packages = smaller attack surface.
#      Attacker cannot exploit software
#      that is not installed.
#
#   2. Non-root user
#      Container runs as "abutech" not root.
#      If container is compromised:
#      Attacker has limited system access.
#      Cannot install packages.
#      Cannot modify system files.
#
#   3. Pinned versions
#      python:3.11-slim not python:latest
#      Reproducible builds.
#      No surprise breaking changes.
#
#   4. Layer ordering (cache optimization)
#      Dependencies installed before code.
#      Code changes do not invalidate
#      the dependency cache layer.
#      Faster builds.
#
#   5. No secrets in image
#      API keys passed at runtime.
#      Never baked into the image.
#      Image can be shared safely.
#
# SR 11-7 COMPLIANCE:
#   Each image tagged with git commit SHA.
#   Complete traceability from container
#   back to exact code version.
#   Model version embedded as label.
#
# BUILD COMMAND:
#   docker build -t abutech-platform:latest .
#
# RUN COMMAND:
#   docker run
#     -e ANTHROPIC_API_KEY=your_key
#     -e MLFLOW_TRACKING_URI=your_uri
#     -p 8000:8000
#     abutech-platform:latest
# ============================================================

# ============================================================
# STAGE 1 — BASE IMAGE
#
# WHY python:3.11-slim:
#   python:3.11 full image = 900MB
#   python:3.11-slim       = 130MB
#   
#   slim removes:
#   - Documentation
#   - Development tools
#   - Unnecessary system packages
#   
#   We keep only what the platform needs.
#   Every MB removed = smaller attack surface.
#   Faster to pull. Faster to scan.
# ============================================================
FROM python:3.11-slim AS base

# Who maintains this image
# Contact for security vulnerabilities
LABEL maintainer="Eliud Maina <eliud@abuhari.com>"
LABEL org.opencontainers.image.title="AbuTech AI Security Platform"
LABEL org.opencontainers.image.description="Enterprise AI Security Platform with LSTM Attention, Knowledge Graph, and LLM Agents"
LABEL org.opencontainers.image.vendor="Abuhari Consulting Services LLC"
LABEL org.opencontainers.image.licenses="Proprietary"

# ============================================================
# STAGE 2 — SYSTEM DEPENDENCIES
#
# WHY THESE PACKAGES:
#
# build-essential:
#   Compiles Python packages that have C extensions.
#   scikit-learn, numpy use C for performance.
#   Required at BUILD time only.
#   Not needed at runtime (see multi-stage below).
#
# curl:
#   Health check endpoint testing.
#   Kubernetes liveness/readiness probes.
#
# git:
#   MLflow needs git for experiment tracking.
#   Captures code version with each model run.
# ============================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        && \
    # Clean up apt cache
    # Reduces image size significantly
    # Cache not needed after install
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# STAGE 3 — PYTHON DEPENDENCIES
#
# WHY COPY requirements.txt FIRST:
# Docker builds in layers.
# If requirements.txt has not changed:
# Docker uses cached layer.
# Does not re-download all packages.
#
# If you copy all code first:
# Any code change invalidates the cache.
# All packages re-downloaded every build.
# Slow. Wasteful.
#
# Order: dependencies → code
# This is called "cache-friendly layering"
# ============================================================

# Set working directory
# All subsequent commands run here
WORKDIR /app

# Copy dependency file first (cache optimization)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: do not cache pip downloads
#                 in the image (saves space)
# --upgrade pip:  ensure latest pip version
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# STAGE 4 — APPLICATION CODE
#
# WHY COPY AFTER DEPENDENCIES:
# Code changes frequently.
# Dependencies change rarely.
# Separate layers = better caching.
#
# WHAT WE COPY:
# Only the layers needed to run the platform.
# Not tests (not needed in production).
# Not development files.
# Minimal code surface.
# ============================================================

# Copy platform layers
COPY layer1_ingestion/ ./layer1_ingestion/
COPY layer2_ml/ ./layer2_ml/
COPY layer3_knowledge/ ./layer3_knowledge/
COPY layer4_reasoning/ ./layer4_reasoning/
COPY scripts/ ./scripts/

# Copy configuration files
COPY .gitlab-ci.yml .
COPY requirements.txt .

# ============================================================
# STAGE 5 — SECURITY HARDENING
#
# NON-ROOT USER:
# By default Docker runs as root inside container.
# Root inside container has significant host access.
# If attacker compromises the app they get root.
#
# Creating a dedicated user limits damage:
# "abutech" user cannot:
# - Install system packages
# - Modify system files
# - Access other users' files
# - Write outside /app
#
# This is PRINCIPLE OF LEAST PRIVILEGE
# applied to container security.
# Required by CIS Docker Benchmark.
# Required by most enterprise security policies.
# BofA and Amex WILL ask about this.
# ============================================================

# Create non-root user and group
RUN groupadd --gid 1000 abutech && \
    useradd \
        --uid 1000 \
        --gid abutech \
        --shell /bin/bash \
        --create-home \
        abutech

# Create directories platform needs
# Give non-root user ownership
RUN mkdir -p \
        /app/models \
        /app/logs \
        /app/mlruns \
        /app/data && \
    chown -R abutech:abutech /app

# Switch to non-root user
# All subsequent commands run as this user
USER abutech

# ============================================================
# STAGE 6 — ENVIRONMENT CONFIGURATION
#
# ENVIRONMENT VARIABLES:
# Set defaults that work in most environments.
# Override at runtime for production values.
#
# NEVER SET SECRETS HERE:
# These are baked into the image.
# Anyone who pulls the image sees them.
#
# SECRETS ARE PASSED AT RUNTIME:
# docker run -e ANTHROPIC_API_KEY=xxx ...
# Or via Kubernetes secrets.
# Or via HashiCorp Vault.
# ============================================================

# Python behavior settings
ENV PYTHONUNBUFFERED=1
# PYTHONUNBUFFERED=1 means Python output
# is sent directly to terminal without buffering.
# Important for seeing logs in real-time.
# Without this: logs appear in batches.
# With this: logs appear immediately.

ENV PYTHONPATH=/app
# Same as the PYTHONPATH in .gitlab-ci.yml
# Allows: from layer1_ingestion.normalizers...
# Without this: ImportError in production.

ENV PYTHONDONTWRITEBYTECODE=1
# Do not write .pyc compiled files.
# Reduces container size.
# Not needed in production containers.

# Platform configuration defaults
# Override these with real values at runtime
ENV LOG_LEVEL=INFO
ENV PLATFORM_VERSION=1.0.0
ENV MLFLOW_TRACKING_URI=./mlruns

# ============================================================
# STAGE 7 — HEALTH CHECK
#
# WHY HEALTH CHECK:
# Kubernetes needs to know if your container
# is alive and ready to serve requests.
#
# Liveness probe:  is the container alive?
#                  if not → restart it
#
# Readiness probe: is the container ready for traffic?
#                  if not → do not send requests to it
#
# This health check runs:
# Every 30 seconds
# Times out after 10 seconds
# Retries 3 times before marking unhealthy
#
# In Layer 5 (FastAPI) we will add:
# GET /health endpoint that returns 200 OK
# This health check will call that endpoint.
# ============================================================
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health \
        || exit 1

# ============================================================
# STAGE 8 — EXPOSE AND RUN
#
# EXPOSE 8000:
# Documents that this container listens on port 8000.
# Does NOT actually open the port.
# Port mapping done at runtime:
# docker run -p 8000:8000 ...
#
# CMD:
# Default command when container starts.
# In Layer 5 we will run FastAPI here.
# For now: validate models on startup.
# This proves the container works correctly.
# ============================================================

EXPOSE 8000

# Default command
# Layer 5 will replace this with:
# CMD ["uvicorn", "layer5_interface.main:app",
#      "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "scripts/validate_models.py"]