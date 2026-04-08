FROM python:3.11-slim

# HuggingFace Spaces metadata
LABEL org.opencontainers.image.title="DataQuality OpenEnv"
LABEL org.opencontainers.image.description="Real-world data quality triage environment for AI agents"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY dataquality_env/ ./dataquality_env/
# --- CHANGED LINE BELOW ---
COPY inference.py . 
# --------------------------
COPY openenv.yaml .
COPY baseline/ ./baseline/
COPY tests/ ./tests/
COPY README.md .

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# --- CHANGED CMD BELOW ---
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
