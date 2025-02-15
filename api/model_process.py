#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Author:      Kuro
# @Time:        1/18/2025 11:10 AM

import subprocess
from concurrent.futures import ProcessPoolExecutor
import asyncio
import cv2
import torch
import torchaudio
from deepface import DeepFace

from api.config import Config
from api.utils import find_three_spoken_digits


def extract_audio(ffmpeg_path, video_path, output_audio_path, sample_rate):
    """
    Extracts audio from the input video using ffmpeg and saves as a WAV file.
    """
    command = [
        ffmpeg_path,
        "-y",  # Overwrite output file if it exists
        "-i", video_path,
        "-vn",  # disable video
        "-t", "15",  # Extract only the first 15 seconds
        "-acodec", "pcm_s16le",  # 16-bit WAV
        "-ar", str(sample_rate),
        "-ac", "1",  # mono
        output_audio_path,
    ]
    subprocess.run(command, check=True)


def detect_face_similarity(video_path, face_model, distance_metric, threshold, reference_encoding):
    """
    Detects faces in the video and checks if there's a match
    with the reference face within a limited number of frames.
    Returns True if matched frames exceed threshold, otherwise False.
    """
    is_match = False
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_count_matching = 0
    spoofing_count = 0

    try:
        while True:
            if frame_count_matching >= 10:
                is_match = True
                break

            if frame_count >= 100:
                break

            ret, frame = video_capture.read()
            if not ret:
                break

            try:
                embedding_faces = DeepFace.represent(frame, model_name=face_model, enforce_detection=False, anti_spoofing=True)
                for face in embedding_faces:
                    if face["face_confidence"] < 0.7:
                        continue
                    result = DeepFace.verify(img1_path=reference_encoding, img2_path=face["embedding"], enforce_detection=False, distance_metric=distance_metric, model_name=face_model,
                                             threshold=threshold, anti_spoofing=True)
                    if result["verified"]:
                        frame_count_matching += 1
            except ValueError as e:
                if "Spoof detected" in str(e):
                    spoofing_count += 1
                if spoofing_count > 6:
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return False, "Spoof detected"

            # # Show the frame
            # cv2.imshow("Video", frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
            frame_count += 1

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

    async def run(self):
        loop = asyncio.get_running_loop()
        try:
            with ProcessPoolExecutor() as pool:
                face_future = loop.run_in_executor(
                    pool,
                    detect_face_similarity,
                    self.video_path,
                    self.face_model,
                    self.distance_metric,
                    self.threshold,
                    self.reference_encoding
                )
                audio_future = loop.run_in_executor(
                    pool,
                    extract_audio,
                    self.ffmpeg_path,
                    self.video_path,
                    self.output_audio_path,
                    self.sample_rate
                )

                # Await audio extraction completion before transcription
                await audio_future

                transcribe_future = loop.run_in_executor(
                    pool,
                    transcribe_long_audio,
                    self.processor,
                    self.model,
                    self.output_audio_path,
                    30,
                    self.sample_rate
                )

                # Await results directly:
                is_match, msg = await face_future
                if msg == "Spoof detected":
                    return {"detail": msg}
                transcription = await transcribe_future
                digits = find_three_spoken_digits(transcription)
                return {"3-digit": digits, "similarity": is_match}
        except Exception as e:
            # Log the exception to understand what might be going wrong
            print(f"Exception in run pipeline: {e}")
            raise
