FROM python:3.11-slim

# System deps for PDF rendering
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Create state dir with correct ownership
RUN mkdir -p /data/state && chown -R appuser:appuser /data

USER appuser
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.main"]
