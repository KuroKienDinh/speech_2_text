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
    rm -rf /var/lib/apt/lists/* \


# Create and switch to a working directory
WORKDIR /app

# Copy requirements.txt from the build context into the container
# The path "api/requirements.txt" works because our build context is the project root
# and the file is located in the api/ subfolder
COPY ./api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire `api` folder into the container
COPY . /app

# Expose the port
EXPOSE 8008

# Run the FastAPI application via python -m
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8008"]
