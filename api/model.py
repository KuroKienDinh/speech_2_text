#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Author:      Kuro
# @Time:        1/18/2025 11:10 AM

import os
import subprocess

import cv2
import face_recognition
import torchaudio

from api.config import Config
from api.utils import find_three_spoken_digits


# ------------------------------------------------
# Class: MediaProcessor
# ------------------------------------------------
class MediaProcessor:
    def __init__(self, video_path: str, reference_image_path: str, media_processor, model, output_audio_path: str = "output_audio.wav", ffmpeg_path: str = Config.ffmpeg_path, threshold: float = Config.threshold,
                 sample_rate: int = Config.sample_rate):
        """
        :param video_path: Path to the input video file.
        :param reference_image_path: Path to the reference image (face to compare).
        :param output_audio_path: Path to store the extracted WAV audio.
        :param threshold: Face distance threshold (lower distance = more similar).
        :param sample_rate: Sample rate in Hz for extracted audio.
        :param ffmpeg_path: Path to ffmpeg binary (adjust if ffmpeg is not in PATH).
        """

        self.video_path = video_path
        self.reference_image_path = reference_image_path
        self.output_audio_path = output_audio_path
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.ffmpeg_path = ffmpeg_path

        # Load face recognition model data early (e.g., reference encodings)
        self.reference_encoding = self._load_reference_face_encoding()

        # Prepare language model processor & model for transcription
        self.processor = media_processor
        self.model = model

    def _load_reference_face_encoding(self):
        """
        Loads and encodes the reference image. Raises ValueError if no face found.
        """
        reference_image = face_recognition.load_image_file(self.reference_image_path)
        encodings = face_recognition.face_encodings(reference_image)

        if len(encodings) == 0:
            raise ValueError(f"No face found in reference image: {self.reference_image_path}")
        return encodings[0]

    def extract_audio(self):
        """
        Extracts audio from the input video using ffmpeg and saves as a WAV file.
        """
        command = [
            self.ffmpeg_path,
            "-y",  # Overwrite output file if it exists
            "-i", self.video_path,
            "-vn",  # disable video
            "-acodec", "pcm_s16le",  # 16-bit WAV
            "-ar", str(self.sample_rate),
            "-ac", "1",  # mono
            self.output_audio_path,
        ]
        subprocess.run(command, check=True)

    def transcribe_audio(self) -> str:
        """
        Transcribes extracted audio using Seamless M4T v2 model.
        Returns the text transcription.
        """
        if not os.path.exists(self.output_audio_path):
            raise FileNotFoundError(f"Extracted audio not found at {self.output_audio_path}. " f"Did you run extract_audio() first?")

        audio, orig_freq = torchaudio.load(self.output_audio_path)

        # Resample if needed
        if orig_freq != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

        inputs = self.processor(audios=audio, sampling_rate=16000, return_tensors="pt")
        output_tokens = self.model.generate(**inputs, tgt_lang="eng", generate_speech=False)
        transcription = self.processor.decode(
            output_tokens[0].tolist()[0], skip_special_tokens=True
        )
        return transcription

    def detect_face_similarity(self) -> bool:
        """
        Detects faces in the video and checks if there's a match
        with the reference face within a limited number of frames.
        Returns True if matched frames exceed threshold, otherwise False.
        """
        video_capture = cv2.VideoCapture(self.video_path)

        frame_count_matching = 0
        is_match = False

        while True:
            if frame_count_matching >= 10:
                is_match = True
                break

            ret, frame = video_capture.read()
            if not ret:
                # End of video
                break

            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings_in_frame:
                face_distance = face_recognition.face_distance([self.reference_encoding], face_encoding)[0]
                if face_distance < self.threshold:
                    frame_count_matching += 1

        video_capture.release()
        cv2.destroyAllWindows()

        return is_match

    def run(self) -> dict:
        """
        Full pipeline: extract audio -> transcribe -> detect face.
        Returns a dict with '3-digit' and 'similarity' keys.
        """
        self.extract_audio()
        transcription = self.transcribe_audio()
        detected_digits = find_three_spoken_digits(transcription)
        is_match = self.detect_face_similarity()
        return {"3-digit": detected_digits, "similarity": is_match}
