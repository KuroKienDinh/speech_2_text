# syntax=docker/dockerfile:1
FROM python:3.12.8

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
        cmake \
        libgtk-3-dev \
        cargo \
        libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

# Create and switch to a working directory
WORKDIR /app
ENV PYTHONPATH=/app
# Copy requirements.txt from the build context into the container
COPY ./api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire context into /app
COPY . /app
# Debug: List files to confirm structure
RUN find /app -type f
# Expose the port
EXPOSE 8008

# Run the FastAPI application via python -m
CMD [ "python", "-m", "api.main" ]
