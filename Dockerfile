FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create temp directory for PBIX extraction
RUN mkdir -p /tmp/pbix_extract

# Copy application files
COPY main.py /app/
COPY config.yml /app/
COPY test.py /app/

# Expose API port
EXPOSE 8000

# Set environment variable to run in non-interactive mode
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]