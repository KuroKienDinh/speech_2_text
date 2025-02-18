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
    global media_model
    try:
        print("🔄 Loading processor and model...")
        media_model["processor"] = AutoProcessor.from_pretrained(Config.audio_model)
        media_model["model"] = SeamlessM4Tv2Model.from_pretrained(Config.audio_model)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        media_model["processor"] = None
        media_model["model"] = None

    # Give some time for the model to load before first request
    await asyncio.sleep(1)


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

    # Ensure models are available
    processor = media_model.get("processor")
    model = media_model.get("model")

    if processor is None or model is None:
        raise HTTPException(status_code=500, detail="Processor or model not loaded. Please restart the server.")

    # Generate unique filenames
    video_ext = os.path.splitext(video_file.filename)[1]
    temp_video_path = f"temp_video_{uuid.uuid4()}{video_ext}"
    img_ext = os.path.splitext(reference_image.filename)[1]
    temp_img_path = f"temp_ref_{uuid.uuid4()}{img_ext}"
    temp_audio_path = f"temp_output_audio_{uuid.uuid4()}.wav"

    try:
        # Save video file
        async with aiofiles.open(temp_video_path, "wb") as out_video
