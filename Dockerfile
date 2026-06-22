# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (skip heavy Jupyter/JupyterLite/GPU packages for slim server image)
RUN grep -v -i 'jupyterlite\|jupyter-core\|notebook\|micropip\|matplotlib\|seaborn\|plotly\|xgboost\|scipy\|torch\|nvidia' requirements.txt > requirements-slim.txt && \
    pip install --no-cache-dir -r requirements-slim.txt || \
    pip install --no-cache-dir -r requirements-slim.txt --no-deps

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app/

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/uploads/research && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
