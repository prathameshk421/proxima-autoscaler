FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install CPU-only PyTorch first (avoids ~1.3GB CUDA downloads), then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

COPY app.py .
COPY openenv.yaml .

COPY checkpoints/ ./checkpoints/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
