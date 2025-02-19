import os
import uuid
import warnings

import aiofiles
from celery.result import AsyncResult
from fastapi import FastAPI, File, UploadFile, HTTPException

from api.celery_app import celery_app
from api.config import Config
from api.tasks import process_media_task  # Import the Celery task

warnings.filterwarnings("ignore")

app = FastAPI(title="Media Processing API", version="1.0.0")
# Use the shared directory
TEMP_DIR = "/app/shared_temp"  # Ensure this matches the mount path in docker-compose.yml
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/process")
async def process_video(
    video_file: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    ffmpeg_path: str = Config.ffmpeg_path,
    threshold: float = Config.threshold,
    sample_rate: int = Config.sample_rate
):
    # Generate unique filenames in the shared directory
    video_ext = os.path.splitext(video_file.filename)[1]
    temp_video_path = os.path.join(TEMP_DIR, f"temp_video_{uuid.uuid4()}{video_ext}")

    img_ext = os.path.splitext(reference_image.filename)[1]
    temp_img_path = os.path.join(TEMP_DIR, f"temp_ref_{uuid.uuid4()}{img_ext}")

    temp_audio_path = os.path.join(TEMP_DIR, f"temp_output_audio_{uuid.uuid4()}.wav")

    try:
        # Save video file asynchronously
        async with aiofiles.open(temp_video_path, "wb") as out_video:
            while True:
                chunk = await video_file.read(1024 * 1024)  # Read in chunks of 1MB
                if not chunk:
                    break
                await out_video.write(chunk)

        # Save reference image asynchronously
        async with aiofiles.open(temp_img_path, "wb") as out_img:
            while True:
                chunk = await reference_image.read(1024 * 1024)
                if not chunk:
                    break
                await out_img.write(chunk)

        # Submit the Celery task
        task = process_media_task.delay(
            temp_video_path,
            temp_img_path,
            temp_audio_path,
            threshold,
            sample_rate,
            ffmpeg_path
        )
        return {"task_id": task.id, "status": "Processing initiated."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/tasks/{task_id}')
def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.state == 'PENDING':
        return {"task_id": task_id, "status": "Pending"}
    elif task_result.state == 'STARTED':
        return {"task_id": task_id, "status": "Processing"}
    elif task_result.state == 'SUCCESS':
        return {"task_id": task_id, "status": "Success", "result": task_result.result}
    elif task_result.state == 'FAILURE':
        return {"task_id": task_id, "status": "Failure", "error": str(task_result.result)}
    else:
        return {"task_id": task_id, "status": str(task_result.state)}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8008)

# curl -X POST http://127.0.0.1:8008/process -F "video_file=@2.webm" -F "reference_image=@2.jpg"
# curl -X GET http://127.0.0.1:8008/tasks/f0aaab5e-2775-4e0c-9d2b-7e39a5eec345
