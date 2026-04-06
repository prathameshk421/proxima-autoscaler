FROM python:3.11-slim

# OS-level dependencies
RUN apt-get update && apt-get install -y gcc git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Enable pipeline layer cache leveraging by explicitly installing requirements first
COPY pyproject.toml .
# Install base project
RUN pip install --no-cache-dir -e .

# Load up actual core logic maps and trained artifacts
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY app.py .

# Standard hugging face space open port assignment
EXPOSE 7860

# Init uvicorn on all network hostings natively routed toward the 7860 config
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
