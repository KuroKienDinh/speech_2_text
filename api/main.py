import os
import uuid
import time
import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from celery.result import AsyncResult
from api.celery_worker import process_media_task

app = FastAPI(title="Media Processing API with Queue", version="1.0.0")


@app.post("/process")
async def process_video(video_file: UploadFile = File(...), reference_image: UploadFile = File(...), ffmpeg_path: str = "ffmpeg", threshold: float = 0.5, sample_rate: int = 16000):
    # Generate temporary filenames
    video_ext = os.path.splitext(video_file.filename)[1]
    temp_video_path = f"temp_video_{uuid.uuid4()}{video_ext}"
    img_ext = os.path.splitext(reference_image.filename)[1]
    temp_img_path = f"temp_ref_{uuid.uuid4()}{img_ext}"
    temp_audio_path = f"temp_output_audio_{uuid.uuid4()}.wav"

    try:
        # Save video asynchronously
        async with aiofiles.open(temp_video_path, "wb") as out_video:
            while chunk := await video_file.read(1024 * 1024):
                await out_video.write(chunk)

        # Save image asynchronously
        async with aiofiles.open(temp_img_path, "wb") as out_img:
            while chunk := await reference_image.read(1024 * 1024):
                await out_img.write(chunk)

        # Submit the task to Celery queue
        task = process_media_task.apply_async(args=[temp_video_path, temp_img_path, temp_audio_path, threshold, sample_rate, ffmpeg_path])

        return {"task_id": task.id, "status": "processing"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=process_media_task)

    if task_result.state == "PENDING":
        return {"task_id": task_id, "status": "waiting"}
    elif task_result.state == "STARTED":
        return {"task_id": task_id, "status": "processing"}
    elif task_result.state == "SUCCESS":
        return {"task_id": task_id, "status": "completed", "result": task_result.result}
    elif task_result.state == "FAILURE":
        return {"task_id": task_id, "status": "failed", "error": str(task_result.result)}

    return {"task_id": task_id, "status": task_result.state}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8008)

# curl -X POST http://127.0.0.1:8008/process -F "video_file=@2.webm" -F "reference_image=@2.jpg"
