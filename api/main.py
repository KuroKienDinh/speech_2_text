import os
import time
import uuid
import warnings
import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from transformers import AutoProcessor, SeamlessM4Tv2Model

from api.config import Config
from api.model_process import MediaProcessor

warnings.filterwarnings("ignore")

# FastAPI Application
app = FastAPI(title="Media Processing API", version="1.0.0")

# Dictionary to store media models
media_model = {}

# Fake database to store task statuses
fake_db = {}


# Lifespan: Manages ProcessPoolExecutor
@asynccontextmanager
async def lifespan(app: FastAPI):
    q = asyncio.Queue()
    pool = ProcessPoolExecutor()  # Process pool for CPU-bound tasks
    asyncio.create_task(process_requests(q, pool))  # Start background task
    yield {"q": q, "pool": pool}
    pool.shutdown()  # Clean up process pool


# Initialize FastAPI with the lifespan manager
app = FastAPI(lifespan=lifespan)


@app.on_event("startup")
async def startup_event():
    """
    Load the large model/processor once at startup.
    """
    media_model["processor"] = AutoProcessor.from_pretrained(Config.audio_model)
    media_model["model"] = SeamlessM4Tv2Model.from_pretrained(Config.audio_model)


@app.post("/process")
async def process_video(
        request: Request,
        video_file: UploadFile = File(...),
        reference_image: UploadFile = File(...),
        ffmpeg_path: str = Config.ffmpeg_path,
        threshold: float = Config.threshold,
        sample_rate: int = Config.sample_rate,
):
    start_time = time.time()

    # Generate unique filenames
    video_ext = os.path.splitext(video_file.filename)[1]
    temp_video_path = f"temp_video_{uuid.uuid4()}{video_ext}"
    img_ext = os.path.splitext(reference_image.filename)[1]
    temp_img_path = f"temp_ref_{uuid.uuid4()}{img_ext}"
    temp_audio_path = f"temp_output_audio_{uuid.uuid4()}.wav"

    try:
        # Save video file
        async with aiofiles.open(temp_video_path, "wb") as out_video:
            while chunk := await video_file.read(1024 * 1024):
                await out_video.write(chunk)

        # Save reference image
        async with aiofiles.open(temp_img_path, "wb") as out_img:
            while chunk := await reference_image.read(1024 * 1024):
                await out_img.write(chunk)

        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        fake_db[task_id] = "Pending..."

        # Add the task to the queue
        request.state.q.put_nowait((task_id, temp_video_path, temp_img_path, temp_audio_path, threshold, sample_rate, ffmpeg_path))

        return {"task_id": task_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def check_status(task_id: str):
    if task_id in fake_db:
        return {"status": fake_db[task_id]}
    else:
        return JSONResponse("Task ID Not Found", status_code=404)


async def process_requests(q: asyncio.Queue, pool: ProcessPoolExecutor):
    """
    Background task to process requests from the queue.
    """
    while True:
        task_id, video_path, img_path, audio_path, threshold, sample_rate, ffmpeg_path = await q.get()
        try:
            processor = MediaProcessor(
                media_processor=media_model["processor"],
                model=media_model["model"],
                video_path=video_path,
                reference_image_path=img_path,
                output_audio_path=audio_path,
                threshold=threshold,
                sample_rate=sample_rate,
                ffmpeg_path=ffmpeg_path
            )

            # Run processing in the process pool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(pool, processor.run)

            # Update task status
            fake_db[task_id] = result
        except Exception as e:
            fake_db[task_id] = f"Error: {str(e)}"
        finally:
            # Cleanup temporary files
            for path in [video_path, img_path, audio_path]:
                if os.path.exists(path):
                    os.remove(path)

# curl -X POST http://127.0.0.1:8008/process -F "video_file=@2.webm" -F "reference_image=@2.jpg"

