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

from transformers import AutoProcessor, SeamlessM4Tv2Model

from api.config import Config
from api.model_process import MediaProcessor

warnings.filterwarnings("ignore")

# FastAPI Application
app = FastAPI(title="Media Processing API", version="1.0.0")

# Dictionary to store media models
media_model = {}

# Constants
MAX_CONCURRENT_TASKS = 3  # Limit concurrent processing tasks
queue = asyncio.Queue()  # Global queue for processing
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)  # Limit concurrency


# Lifespan: Manages ProcessPoolExecutor
@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = ProcessPoolExecutor()  # Global process pool for CPU tasks
    asyncio.create_task(process_requests(queue, pool))  # Start background task
    yield {"pool": pool}
    pool.shutdown()  # Cleanup on shutdown


# Initialize FastAPI with lifespan
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

        # Create a future object to wait for the result
        future = asyncio.Future()

        # Create task entry
        task = {
            "video_path": temp_video_path,
            "img_path": temp_img_path,
            "audio_path": temp_audio_path,
            "threshold": threshold,
            "sample_rate": sample_rate,
            "ffmpeg_path": ffmpeg_path,
            "processor": media_model["processor"],
            "model": media_model["model"],
            "semaphore": semaphore,  # Ensure proper concurrency handling
            "future": future  # Store future to retrieve result later
        }

        # Put task in queue
        await queue.put(task)

        # Wait for result
        result = await future

        end_time = time.time()
        result["elapsed_time"] = end_time - start_time
        return result  # Return processed output immediately

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp files
        for path in [temp_video_path, temp_img_path, temp_audio_path]:
            if os.path.exists(path):
                os.remove(path)


async def process_requests(q: asyncio.Queue, pool: ProcessPoolExecutor):
    """
    Background task to process requests from the queue.
    - Ensures only MAX_CONCURRENT_TASKS requests are processed at once.
    - Requests will wait in the queue if all slots are full.
    """
    while True:
        task = await q.get()
        async with task["semaphore"]:  # Wait if the maximum limit is reached
            try:
                # Initialize processor
                processor = MediaProcessor(
                    media_processor=task["processor"],
                    model=task["model"],
                    video_path=task["video_path"],
                    reference_image_path=task["img_path"],
                    output_audio_path=task["audio_path"],
                    threshold=task["threshold"],
                    sample_rate=task["sample_rate"],
                    ffmpeg_path=task["ffmpeg_path"]
                )

                # Run processing in ProcessPoolExecutor
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(pool, processor.run)

                # Set the result for the waiting request
                task["future"].set_result(result)

            except Exception as e:
                task["future"].set_exception(e)

            finally:
                q.task_done()  # Mark task as completed
