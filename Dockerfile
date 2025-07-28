FROM --platform=linux/amd64 python:3.10-slim as base

WORKDIR /app

# Install system dependencies for PyMuPDF and scikit-learn
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        python3-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for Challenge 1A
COPY requirements-1a.txt ./
RUN pip install --no-cache-dir -r requirements-1a.txt

# Copy only the files needed for Challenge 1A
COPY outline_extractor/ ./outline_extractor/
COPY process_pdfs.py ./
COPY Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/ Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/

# Set up entrypoint for processing
ENTRYPOINT ["python", "process_pdfs.py"]