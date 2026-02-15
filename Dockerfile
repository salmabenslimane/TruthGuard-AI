FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install packages individually to see which fails
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p /app/models /app/logs

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models/cache

CMD ["python", "src/train.py"]