FROM python:3.11-slim

WORKDIR /app

# Optimize Python execution and disable pip cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (optional, but good for some python libs)
# We keep it slim. If pandas needs build tools, we might need gcc but binary wheels usually work.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create outputs dir for mapping
RUN mkdir -p /app/outputs

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
