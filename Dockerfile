FROM python:3.11-slim

# OS-level dependencies (include git-lfs for LFS-tracked checkpoints)
RUN apt-get update && apt-get install -y gcc git git-lfs && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# Copy source and config for install
COPY pyproject.toml .
COPY src/ ./src/

# Install project (non-editable; hatchling needs src/ present)
RUN pip install --no-cache-dir .

# Copy entrypoint
COPY app.py .

# Copy checkpoints — on HF Spaces these may be LFS pointers,
# so we also try fetching real content via git lfs if .git is available.
COPY checkpoints/ ./checkpoints/
RUN if [ -d "/app/.git" ]; then \
      cd /app && git lfs pull; \
    fi

# Standard hugging face space open port assignment
EXPOSE 7860

# Init uvicorn on all network hostings natively routed toward the 7860 config
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
