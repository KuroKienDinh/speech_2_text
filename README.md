# Media Processing API

This project provides an API service that processes video submissions with face recognition and audio transcription capabilities, using AI models. The system is containerized with Docker and uses FastAPI for the REST API and Celery for asynchronous task processing.

## Features

- Face similarity verification between reference image and video
- Anti-spoofing detection for face recognition
- Audio extraction and transcription from videos
- Three-digit code detection from spoken words
- Asynchronous processing using Celery task queue
- GPU acceleration support

## Architecture

The system consists of three main components:
- **FastAPI Application**: Handles API requests and file uploads
- **Celery Worker**: Processes media files asynchronously
- **Redis**: Message broker for Celery tasks

## Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for GPU acceleration)
- FFmpeg

## Installation

1. Clone the repository
2. Start the services using Docker Compose:

```bash
docker compose -f docker-compose-gpu.yml -p media_process up -d --build
```

To stop and remove containers:

```bash
docker compose -f docker-compose-gpu.yml -p media_process down -v
```

## API Endpoints

### Process Video

```
POST /process
```

**Parameters**:
- `video_file`: Video file to be processed
- `reference_image`: Reference image for face comparison
- `ffmpeg_path` (optional): Path to FFmpeg executable
- `threshold` (optional): Face recognition similarity threshold
- `sample_rate` (optional): Audio sample rate

**Example**:
```bash
curl -X POST http://localhost:8008/process -F "video_file=@video.webm" -F "reference_image=@photo.jpg"
```

**Response**:
```json
{
  "task_id": "abc123",
  "status": "Processing initiated."
}
```

### Task Status

```
GET /tasks/{task_id}
```

**Example**:
```bash
curl -X GET http://localhost:8008/tasks/abc123
```

**Response**:
```json
{
  "task_id": "abc123",
  "status": "Success",
  "result": {
    "3-digit": "123",
    "similarity": true
  }
}
```

## Technology Stack

- Python 3.x
- FastAPI
- Celery
- Redis
- TensorFlow 2.12.0
- PyTorch
- DeepFace
- Transformers (Hugging Face)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
