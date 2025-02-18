#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Author:      Kuro
# @Time:        1/18/2025 11:10 AM

import subprocess
import cv2
import torch
import torchaudio
from deepface import DeepFace

from api.config import Config
from api.utils import find_three_spoken_digits

def extract_audio(ffmpeg_path, video_path, output_audio_path, sample_rate):
    """
    Extract audio from the video using ffmpeg.
    """
    command = [
        ffmpeg_path,
        "-y",              # Overwrite output if exists
        "-i", video_path,
        "-vn",             # Disable video
        "-t", "15",        # Extract first 15 seconds
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",        # Mono audio
        output_audio_path,
    ]
    subprocess.run(command, check=True)

def detect_face_similarity(video_path, face_model, distance_metric, threshold, reference_encoding):
    """
    Detect face similarity in the video using DeepFace.
    Returns a tuple (is_match, message). If too many spoof attempts occur, returns ("Spoof detected").
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
                embedding_faces = DeepFace.represent(
                    frame,
                    model_name=face_model,
                    enforce_detection=False,
                    anti_spoofing=True
                )
                for face in embedding_faces:
                    if face["face_confidence"] < 0.7:
                        continue
                    result = DeepFace.verify(
                        img1_path=reference_encoding,
                        img2_path=face["embedding"],
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
                if spoofing_count > 6:
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return False, "Spoof detected"
            frame_count += 1

        video_capture.release()
        cv2.destroyAllWindows()
        return is_match, None
    except Exception as e:
        try:
            video_capture.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        return False, str(e)

def transcribe_long_audio(processor, model, output_audio_path, chunk_length=30, sample_rate=16000):
    """
    Split the audio into chunks and transcribe each one.
    """
    audio, orig_freq = torchaudio.load(output_audio_path)
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
    def __init__(
        self,
        video_path: str,
        reference_image_path: str,
        media_processor,
        model,
        output_audio_path: str = "output_audio.wav",
        ffmpeg_path: str = Config.ffmpeg_path,
        threshold: float = Config.threshold,
        sample_rate: int = Config.sample_rate,
        face_model: str = Config.face_model_name,
        distance_metric = Config.distance_metric
    ):
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

        # Load the reference face encoding during initialization
        self.reference_encoding = self._load_reference_face_encoding()

    def _load_reference_face_encoding(self):
        """
        Load and encode the reference image.
        Raises a ValueError if no face is found.
        """
        encodings = DeepFace.represent(self.reference_image_path, model_name=self.face_model)
        if not encodings or encodings[0]["face_confidence"] < 0.7:
            raise ValueError(f"No face found in reference image: {self.reference_image_path}")
        return encodings[0]["embedding"]

    def run(self):
        """
        Execute the media processing pipeline:
          1. Face similarity detection.
          2. Audio extraction.
          3. Audio transcription.
          4. Extraction of spoken digits.
        """
        try:
            is_match, msg = detect_face_similarity(
                self.video_path, self.face_model, self.distance_metric, self.threshold, self.reference_encoding
            )
            if msg == "Spoof detected":
                return {"detail": msg}

            extract_audio(self.ffmpeg_path, self.video_path, self.output_audio_path, self.sample_rate)

            transcription = transcribe_long_audio(
                self.processor, self.model, self.output_audio_path, chunk_length=30, sample_rate=self.sample_rate
            )

            digits = find_three_spoken_digits(transcription)
            return {"3-digit": digits, "similarity": is_match}
        except Exception as e:
            return {"error": str(e)}
