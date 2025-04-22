FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY app/ /app/
COPY requirements.txt /app/
COPY config.yml /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create temp directory for PBIX extraction
RUN mkdir -p /tmp/pbix_extract

# Expose API port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]