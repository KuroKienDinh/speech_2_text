#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    celery_worker.py
# @Author:      Kuro
# @Time:        2/18/2025 9:04 PM

from celery import Celery
import os
import uuid
import time
import aiofiles
from api.config import Config
from api.model_process import MediaProcessor
from transformers import AutoProcessor, SeamlessM4Tv2Model

# Use the Redis service name from docker-compose.yml instead of localhost
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery("media_processing", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task(bind=True)
def test_task(self):
    return "Celery GPU task is running!"

# Load the large model once globally
media_model = {
    "processor": AutoProcessor.from_pretrained(Config.audio_model),
    "model": SeamlessM4Tv2Model.from_pretrained(Config.audio_model)
}


@celery_app.task(bind=True)
def process_media_task(self, video_path, img_path, output_audio_path, threshold, sample_rate, ffmpeg_path):
    start_time = time.time()

    try:
        processor = MediaProcessor(
            media_processor=media_model["processor"],
            model=media_model["model"],
            video_path=video_path,
            reference_image_path=img_path,
            output_audio_path=output_audio_path,
            threshold=threshold,
            sample_rate=sample_rate,
            ffmpeg_path=ffmpeg_path
        )

        results = processor.run()
        end_time = time.time()
        results["elapsed_time"] = end_time - start_time
        return results

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        # Clean up temp files
        for path in [video_path, img_path, output_audio_path]:
            if os.path.exists(path):
                os.remove(path)


