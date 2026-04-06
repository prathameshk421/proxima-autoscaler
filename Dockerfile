FROM python:3.11-slim

# OS-level dependencies
RUN apt-get update && apt-get install -y gcc git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source and config for install
COPY pyproject.toml .
COPY src/ ./src/

# Install project (non-editable; hatchling needs src/ present)
RUN pip install --no-cache-dir .

# Copy trained artifacts and entrypoint
COPY checkpoints/ ./checkpoints/
COPY app.py .

# Standard hugging face space open port assignment
EXPOSE 7860

# Init uvicorn on all network hostings natively routed toward the 7860 config
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
