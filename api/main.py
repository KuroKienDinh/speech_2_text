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

# Global dictionary to store models
media_model = {}

# Constants for concurrent processing
MAX_CONCURRENT_TASKS = 3
queue = asyncio.Queue()  # Global task queue
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)  # Concurrency limiter

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models here
    try:
        media_model["processor"] = AutoProcessor.from_pretrained(Config.audio_model)
        media_model["model"] = SeamlessM4Tv2Model.from_pretrained(Config.audio_model)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Optionally, you can re-raise the exception to prevent the app from starting without models
        raise e

    pool = ProcessPoolExecutor()  # Global process pool for CPU tasks
    asyncio.create_task(process_requests(queue, pool))  # Start background task
    yield {"pool": pool}
    pool.shutdown()  # Cleanup on shutdown

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)


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

    # Create unique temporary file names
    video_ext = os.path.splitext(video_file.filename)[1]
    temp_video_path = f"temp_video_{uuid.uuid4()}{video_ext}"
    img_ext = os.path.splitext(reference_image.filename)[1]
    temp_img_path = f"temp_ref_{uuid.uuid4()}{img_ext}"
    temp_audio_path = f"temp_output_audio_{uuid.uuid4()}.wav"

    try:
        # Save the uploaded video file
        async with aiofiles.open(temp_video_path, "wb") as out_video:
            while chunk := await video_file.read(1024 * 1024):
                await out_video.write(chunk)

        # Save the uploaded reference image
        async with aiofiles.open(temp_img_path, "wb") as out_img:
            while chunk := await reference_image.read(1024 * 1024):
                await out_img.write(chunk)

        # Retrieve loaded models from the global dictionary
        processor = media_model.get("processor")
        model = media_model.get("model")
        if processor is None or model is None:
            raise HTTPException(status_code=500, detail="Processor or model not loaded.")

        # Create a Future to hold the processing result
        future = asyncio.Future()

        # Build the task entry to be processed
        task = {
            "video_path": temp_video_path,
            "img_path": temp_img_path,
            "audio_path": temp_audio_path,
            "threshold": threshold,
            "sample_rate": sample_rate,
            "ffmpeg_path": ffmpeg_path,
            "processor": processor,
            "model": model,
            "semaphore": semaphore,
            "future": future
        }

        # Put the task into the global queue
        await queue.put(task)

        # Wait until the background processor sets the result
        result = await future

        # Add elapsed time information to the result
        end_time = time.time()
        result["elapsed_time"] = end_time - start_time
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files if they exist
        for path in [temp_video_path, temp_img_path, temp_audio_path]:
            if os.path.exists(path):
                os.remove(path)

async def process_requests(q: asyncio.Queue, pool: ProcessPoolExecutor):
    """
    Background task that processes queued media tasks.
    It uses a semaphore to ensure no more than MAX_CONCURRENT_TASKS run concurrently.
    """
    while True:
        task = await q.get()
        async with task["semaphore"]:
            try:
                media_processor_instance = MediaProcessor(
                    media_processor=task["processor"],
                    model=task["model"],
                    video_path=task["video_path"],
                    reference_image_path=task["img_path"],
                    output_audio_path=task["audio_path"],
                    threshold=task["threshold"],
                    sample_rate=task["sample_rate"],
                    ffmpeg_path=task["ffmpeg_path"]
                )
                loop = asyncio.get_running_loop()
                # Run the processor in a separate process
                result = await loop.run_in_executor(pool, media_processor_instance.run)
                task["future"].set_result(result)
            except Exception as e:
                task["future"].set_exception(e)
            finally:
                q.task_done()
