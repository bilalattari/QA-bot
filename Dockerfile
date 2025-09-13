# Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="/home/appuser/.local/bin:$PATH"

WORKDIR /app

# copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git ca-certificates && \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    apt-get remove -y build-essential gcc && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Create non-root user for safety
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
