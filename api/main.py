#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    main_not_api.py
# @Author:      Kuro
# @Time:        1/18/2025 11:04 AM

import os
import time
import uuid
import warnings

import aiofiles
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.concurrency import run_in_threadpool
from transformers import AutoProcessor, SeamlessM4Tv2Model

from api.config import Config
from api.model import MediaProcessor

warnings.filterwarnings("ignore")

app = FastAPI(title="Media Processing API", version="1.0.0")
media_model = {}


@app.on_event("startup")
async def startup_event():
    """
    Load the large model/processor once at startup.
    """
    media_model["processor"] = AutoProcessor.from_pretrained(Config.audio_model)
    media_model["model"] = SeamlessM4Tv2Model.from_pretrained(Config.audio_model)

@app.post("/process")
async def process_video(video_file: UploadFile = File(...), reference_image: UploadFile = File(...), ffmpeg_path: str = Config.ffmpeg_path, threshold: float = Config.threshold, sample_rate: int = Config.sample_rate):
    start_time = time.time()
    # Save video temporarily
    video_ext = os.path.splitext(video_file.filename)[1]
    temp_video_path = f"temp_video_{uuid.uuid4()}{video_ext}"

    # Save reference image temporarily
    img_ext = os.path.splitext(reference_image.filename)[1]
    temp_img_path = f"temp_ref_{uuid.uuid4()}{img_ext}"

    # Generate unique filename for output audio
    temp_audio_path = f"temp_output_audio_{uuid.uuid4()}.wav"

    try:
        # --- Save video file asynchronously ---
        async with aiofiles.open(temp_video_path, "wb") as out_video:
            # Read file in chunks to avoid loading whole file in memory at once
            while True:
                chunk = await video_file.read(1024 * 1024)  # 1MB per chunk
                if not chunk:
                    break
                await out_video.write(chunk)

        # --- Save reference image asynchronously ---
        async with aiofiles.open(temp_img_path, "wb") as out_img:
            while True:
                chunk = await reference_image.read(1024 * 1024)  # 1MB per chunk
                if not chunk:
                    break
                await out_img.write(chunk)

        # Initialize and run MediaProcessor
        processor = MediaProcessor(media_processor=media_model["processor"], model=media_model["model"], video_path=temp_video_path, reference_image_path=temp_img_path, output_audio_path=temp_audio_path,
                                   threshold=threshold, sample_rate=sample_rate, ffmpeg_path=ffmpeg_path)

        # Run the blocking processor.run() in a threadpool
        results = await run_in_threadpool(processor.run)
        end_time = time.time()
        results["elapsed_time"] = end_time - start_time
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp files
        for path in [temp_video_path, temp_img_path, temp_audio_path]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)

    # curl -X POST http://127.0.0.1:8008/process -F "video_file=@2.webm" -F "reference_image=@2.jpg"
