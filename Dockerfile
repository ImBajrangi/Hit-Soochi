FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY optimizer.py .
COPY server.py .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the server on HF's expected port
CMD ["python", "-c", "import uvicorn; from server import app; uvicorn.run(app, host='0.0.0.0', port=7860)"]
