# docker/Dockerfile
# This file defines how to package our entire application
# into a container — same environment everywhere.
# Docker guarantees it runs identically on your laptop,
# your colleague's machine, and any cloud server.

# Start from official Python 3.11 slim image
# slim = smaller size, only essential OS packages
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies OpenCV needs
# These are OS-level packages, not Python packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer
# So if only your code changes, pip install is skipped
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Expose port 8000 so outside world can reach the API
EXPOSE 8000

# Command that runs when container starts
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]