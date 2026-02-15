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
RUN pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers==4.30.2
RUN pip install --no-cache-dir pandas==2.0.3
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir scikit-learn==1.3.0
RUN pip install --no-cache-dir fastapi==0.100.0
RUN pip install --no-cache-dir uvicorn[standard]==0.23.0
RUN pip install --no-cache-dir pydantic==2.0.3
RUN pip install --no-cache-dir tqdm==4.65.0

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p /app/models /app/logs

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models/cache

CMD ["python", "src/train.py"]