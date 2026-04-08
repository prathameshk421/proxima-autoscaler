FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc git git-lfs && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

RUN pip install --no-cache-dir .

COPY app.py .
COPY inference_llm.py .
COPY openenv.yaml .

COPY checkpoints/ ./checkpoints/
RUN if [ -d "/app/.git" ]; then \
      cd /app && git lfs pull; \
    fi

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
