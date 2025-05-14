# Use official Python 3.10 slim image to match conda environment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pciutils \
    lshw \
    poppler-utils \
    libmagic1 \
    unrtf \
    tesseract-ocr \
    libxml2-dev \
    libxslt1-dev \
    libzbar0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python packages
RUN pip install -r requirements.txt

# (Optional) Install Ollama inside the container
RUN curl -fsSL https://ollama.com/install.sh | sh || echo "Ollama install skipped (for manual/model-based runs)"

# Pull models here if needed (optional; uncomment & replace MODEL_NAME)
# RUN ollama pull llama3

# Copy entire source code into container
COPY . .

# Set up DATA_DOCUMENTS directory
RUN mkdir -p /app/DATA_DOCUMENTS

# Default command to run app (adjust if needed)
CMD ["sh", "command.sh"]
