#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    tasks.py
# @Author:      Kuro
# @Time:        2/19/2025 10:15 AM
import asyncio
# api/tasks.py

import os

from transformers import AutoProcessor, SeamlessM4Tv2Model

from api.config import Config
from .celery_app import celery_app
from .model_process import MediaProcessor

# Global variables to store models
media_model = {}


@celery_app.task(bind=True)
def process_media_task(
    video_path,
    temp_img_path,
    temp_audio_path,
    threshold,
    sample_rate,
    ffmpeg_path
):
    # Ensure models are loaded
    if not media_model:
        media_model["processor"] = AutoProcessor.from_pretrained(Config.audio_model)
        media_model["model"] = SeamlessM4Tv2Model.from_pretrained(Config.audio_model)

    processor = MediaProcessor(
        media_processor=media_model["processor"],
        model=media_model["model"],
        video_path=video_path,
        reference_image_path=temp_img_path,
        output_audio_path=temp_audio_path,
        threshold=threshold,
        sample_rate=sample_rate,
        ffmpeg_path=ffmpeg_path
    )
    try:
        # Check if run() is a coroutine function
        if asyncio.iscoroutinefunction(processor.run):
            # Run the coroutine in an event loop
            results = asyncio.run(processor.run())
        else:
            # If run() is synchronous
            results = processor.run()
        return results
    except Exception as e:
        return {'detail': str(e)}
    finally:
        # Clean up temp files
        for path in [video_path, temp_img_path, temp_audio_path]:
            if os.path.exists(path):
                os.remove(path)


@celery_app.on_after_configure.connect
def setup_models(sender, **kwargs):
    """
    Load models once when the worker starts.
    """
    global media_model
    media_model["processor"] = AutoProcessor.from_pretrained(Config.audio_model)
    media_model["model"] = SeamlessM4Tv2Model.from_pretrained(Config.audio_model)
