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

    def transcribe_audio(self):
        """
        Transcribes extracted audio using Seamless M4T v2 model.
        Returns the text transcription.
        """
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

    def transcribe_long_audio(self, chunk_length=30):
        audio, orig_freq = torchaudio.load(self.output_audio_path)

        # Resample if needed
        if orig_freq != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

        # Convert audio to a NumPy array for easy slicing
        audio_np = audio.squeeze().numpy()

        # Calculate the number of samples in each chunk
        samples_per_chunk = chunk_length * 16000  # 16000 Hz * chunk_length seconds

        # Iterate through chunks
        transcription_list = []
        start = 0
        while start < len(audio_np):
            end = min(start + samples_per_chunk, len(audio_np))
            audio_chunk = audio_np[start:end]

            # Convert back to torch.Tensor
            audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

            inputs = self.processor(
                audios=audio_tensor, sampling_rate=16000, return_tensors="pt"
            )
            output_tokens = self.model.generate(**inputs, tgt_lang="eng", generate_speech=False)
            chunk_text = self.processor.decode(
                output_tokens[0].tolist()[0],
                skip_special_tokens=True
            )
            transcription_list.append(chunk_text)
            start += samples_per_chunk
        return " ".join(transcription_list)

    def detect_face_similarity(self):
        """
        Detects faces in the video and checks if there's a match
        with the reference face within a limited number of frames.
        Returns True if matched frames exceed threshold, otherwise False.
        """
        is_match = False
        video_capture = cv2.VideoCapture(self.video_path)
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
                    embedding_faces = DeepFace.represent(frame, model_name=self.face_model, enforce_detection=False, anti_spoofing=True)
                    for face in embedding_faces:
                        if face["face_confidence"] < 0.7:
                            continue
                        result = DeepFace.verify(img1_path=self.reference_encoding, img2_path=face["embedding"], enforce_detection=False, distance_metric=self.distance_metric, model_name=self.face_model,
                                                 threshold=self.threshold, anti_spoofing=True)
                        if result["verified"]:
                            frame_count_matching += 1
                except ValueError as e:
                    if "Spoof detected" in str(e):
                        spoofing_count += 1
                    if spoofing_count > 10:
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

    def run(self):
        """
        Full pipeline: extract audio -> transcribe -> detect face.
        Returns a dict with '3-digit' and 'similarity' keys.
        """
        self.extract_audio()
        # transcription = self.transcribe_audio()
        transcription = self.transcribe_long_audio()
        detected_digits = find_three_spoken_digits(transcription)
        is_match, msg = self.detect_face_similarity()
        if msg == "Spoof detected":
            return {"detail": msg}
        return {"3-digit": detected_digits, "similarity": is_match}
