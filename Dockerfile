# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Prevents Python from writing .pyc files and makes output unbuffered (good for logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level dependencies commonly needed by:
# - opencv-python (libgl1, libglib2.0-0)
# - Pillow (libjpeg-dev, zlib)
# - cairosvg / reportlab (libcairo2, libpango)
# - pycairo (libcairo2-dev, pkg-config)
# - build-essential for any packages with C-extensions if needed
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libcairo2 \
    libcairo2-dev \
    libpango-1.0-0 \
    libpango1.0-dev \
    libfreetype6 \
    libjpeg-dev \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (helps Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install python deps (no cache to keep image smaller)
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install jupyter notebook

# Copy project source
COPY . .

# Default command - drops you into a shell; change this to your app start command later
CMD ["bash"]
