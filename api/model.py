#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Author:      Kuro
# @Time:        1/18/2025 11:10 AM

import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import torchaudio
from deepface import DeepFace

from api.config import Config
from api.utils import find_three_spoken_digits

executor = ThreadPoolExecutor(max_workers=4)


def extract_audio(ffmpeg_path, video_path, output_audio_path, sample_rate):
    """
    Extracts audio from the input video using ffmpeg and saves as a WAV file.
    """
    command = [
        ffmpeg_path,
        "-y",  # Overwrite output file if it exists
        "-i", video_path,
        "-t", "15",  # Extract only the first 15 seconds
        "-vn",  # disable video
        "-acodec", "pcm_s16le",  # 16-bit WAV
        "-ar", str(sample_rate),
        "-ac", "1",  # mono
        output_audio_path,
    ]
    subprocess.run(command, check=True)


def detect_face_similarity(video_path, face_model, distance_metric, threshold, reference_encoding, batch_size=5):
    """
    Detects faces in the video by processing multiple frames in a batch.
    Uses DeepFace.represent() to extract embeddings for each batch, then
    runs DeepFace.verify() on all embeddings at once for efficiency.

    Returns:
        - True if matched frames exceed threshold.
        - False if a spoofing attack is detected (>6 spoofed frames).
        - None if no error occurs, otherwise returns an error message.
    """
    is_match = False
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_count_matching = 0
    spoofing_count = 0
    frames_buffer = []  # Store frames for batch processing
    try:
        while True:
            if frame_count_matching >= 10:
                is_match = True
                break

            if frame_count >= 100:  # Before we only use first 100 frames
                break

            ret, frame = video_capture.read()
            if not ret:
                break

            frames_buffer.append(frame)  # Collect frames for batching
            frame_count += 1

            # Process when buffer reaches batch size
            if len(frames_buffer) >= batch_size:
                try:
                    # Extract embeddings in batch
                    batch_results = DeepFace.represent(frames_buffer, model_name=face_model, enforce_detection=False, anti_spoofing=True)

                    # Process embeddings for each frame in batch
                    for face_data in batch_results:
                        if face_data["face_confidence"] < 0.7:
                            continue

                        try:
                            # Compare embeddings using DeepFace.verify()
                            result = DeepFace.verify(
                                img1_path=reference_encoding,
                                img2_path=face_data["embedding"],
                                enforce_detection=False,
                                distance_metric=distance_metric,
                                model_name=face_model,
                                threshold=threshold,
                                anti_spoofing=True
                            )

                            if result["verified"]:
                                frame_count_matching += 1

                        except ValueError as e:
                            if "Spoof detected" in str(e):
                                spoofing_count += 1

                    # Clear buffer after processing
                    frames_buffer = []

                    if spoofing_count > 6:
                        video_capture.release()
                        cv2.destroyAllWindows()
                        return False, "Spoof detected"

                except Exception as e:
                    print(f"Error processing batch: {e}")

        # Cleanup
        video_capture.release()
        cv2.destroyAllWindows()
        return is_match, None

    except Exception as e:
        try:
            video_capture.release()
            cv2.destroyAllWindows()
        except:
            pass
        return False, str(e)


def transcribe_long_audio(processor, model, output_audio_path, chunk_length=30, sample_rate=16000):
    """
    Splits audio into chunks and transcribes each chunk.
    """
    audio, orig_freq = torchaudio.load(output_audio_path)
    # Resample if needed
    if orig_freq != sample_rate:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=sample_rate)

    audio_np = audio.squeeze().numpy()
    samples_per_chunk = chunk_length * sample_rate

    transcription_list = []
    start = 0
    while start < len(audio_np):
        end = min(start + samples_per_chunk, len(audio_np))
        audio_chunk = audio_np[start:end]

        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        inputs = processor(audios=audio_tensor, sampling_rate=sample_rate, return_tensors="pt")
        output_tokens = model.generate(**inputs, tgt_lang="eng", generate_speech=False)
        chunk_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        transcription_list.append(chunk_text)
        start += samples_per_chunk

    return " ".join(transcription_list)


class MediaProcessor:
    def __init__(self, video_path: str, reference_image_path: str, media_processor, model, output_audio_path: str = "output_audio.wav", ffmpeg_path: str = Config.ffmpeg_path, threshold: float = Config.threshold,
                 sample_rate: int = Config.sample_rate, face_model: str = Config.face_model_name, distance_metric=Config.distance_metric):
        self.video_path = video_path
        self.reference_image_path = reference_image_path
        self.output_audio_path = output_audio_path
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.ffmpeg_path = ffmpeg_path

        self.processor = media_processor
        self.model = model
        self.face_model = face_model
        self.distance_metric = distance_metric

        # Load face recognition model data early (e.g., reference encodings)
        self.reference_encoding = self._load_reference_face_encoding()

    def _load_reference_face_encoding(self):
        """
        Loads and encodes the reference image. Raises ValueError if no face found.
        """
        encodings = DeepFace.represent(self.reference_image_path, model_name=self.face_model)

        if encodings[0]["face_confidence"] < 0.7:
            raise ValueError(f"No face found in reference image: {self.reference_image_path}")
        return encodings[0]["embedding"]

    def run(self):
        """
        Main pipeline:
          1) Parallel: extract audio & face detection
          2) Transcribe after audio is ready
          3) Collect results
        """
        face_future = executor.submit(detect_face_similarity, self.video_path, self.face_model, self.distance_metric, self.threshold, self.reference_encoding, batch_size=5)
        audio_future = executor.submit(extract_audio, self.ffmpeg_path, self.video_path, self.output_audio_path, self.sample_rate)
        audio_future.result()
        transcribe_future = executor.submit(transcribe_long_audio, self.processor, self.model, self.output_audio_path, 30, self.sample_rate)

        is_match, msg = face_future.result()
        if msg == "Spoof detected":
            return {"detail": msg}
        transcription = transcribe_future.result()
        digits = find_three_spoken_digits(transcription)
        return {"3-digit": digits, "similarity": is_match}
