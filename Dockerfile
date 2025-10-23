FROM python:3.10.13-slim

WORKDIR /app

# Install system dependencies for PDF processing and OCR
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create temp directory for file processing
RUN mkdir -p /tmp

# Expose port
EXPOSE 10000

# Run the application
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
