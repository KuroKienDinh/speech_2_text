#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    main.py
# @Author:      Kuro
# @Time:        1/16/2025 11:17 PM
import argparse
import os
import subprocess
import warnings

import cv2
import face_recognition
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

warnings.filterwarnings("ignore")


def find_three_spoken_digits(text: str) -> str:
    """
    Finds exactly 3 consecutive spoken digits in the text and returns them
    as a string (e.g., "four seven one" => "471"). Returns None if not found.
    """

    # Map of spoken words to digits
    word_to_digit = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9"
    }

    # Tokenize by whitespace and normalize to lowercase
    tokens = text.lower().split()

    # To keep track of consecutive digits
    consecutive_digits = []

    for token in tokens:
        # If token maps to a digit, add it
        if token in word_to_digit:
            consecutive_digits.append(word_to_digit[token])
        else:
            # If the token is not a digit, reset the consecutive list
            consecutive_digits = []

        # If we have exactly 3 digits in a row, return them as a string
        if len(consecutive_digits) == 3:
            return "".join(consecutive_digits)

    # If not found, return None
    return ""


class MediaProcessor:
    def __init__(self, video_path: str, reference_image_path: str, output_audio_path: str = "output_audio.wav", threshold: float = 0.6, sample_rate: int = 16000, ffmpeg_path: str = "ffmpeg", video_ratio: float = 1.0):
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
        self.video_ratio = video_ratio

        # Load face recognition model data early (e.g., reference encodings)
        self.reference_encoding = self._load_reference_face_encoding()

        # Prepare language model processor & model for transcription
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

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
        # Build ffmpeg command
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
        # Run ffmpeg
        # print(f"Extracting audio from {self.video_path} to {self.output_audio_path}...")
        subprocess.run(command, check=True)
        # print("Audio extraction complete.")

    def transcribe_audio(self) -> str:
        """
        Transcribes extracted audio using Seamless M4T v2 model.
        Returns the text transcription.
        """
        if not os.path.exists(self.output_audio_path):
            raise FileNotFoundError(f"Extracted audio not found at {self.output_audio_path}. "
                                    f"Did you run extract_audio() first?")
        # print(f"Loading audio for transcription from {self.output_audio_path}...")
        audio, orig_freq = torchaudio.load(self.output_audio_path)

        # Resample if needed
        if orig_freq != 16000:
            # print(f"Resampling audio from {orig_freq} Hz to 16000 Hz...")
            audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

        # Prepare model input
        # print("Preparing model inputs...")
        inputs = self.processor(audios=audio, sampling_rate=16000, return_tensors="pt")

        # Generate text
        # print("Generating transcription...")
        output_tokens = self.model.generate(**inputs, tgt_lang="eng", generate_speech=False)
        transcription = self.processor.decode(
            output_tokens[0].tolist()[0], skip_special_tokens=True
        )
        return transcription

    def detect_face_similarity(self) -> float:
        """
        Detects face(s) in the video and computes an average face distance
        with respect to the reference face. Returns an average similarity score (1 - avg_distance).
        """
        # print(f"Detecting faces in video (first {self.video_ratio} only): {self.video_path}...")
        video_capture = cv2.VideoCapture(self.video_path)

        frame_count_matching = 0
        is_match = False

        while True:
            # Stop if we have processed
            if frame_count_matching >= 10:
                is_match = True
                break

            ret, frame = video_capture.read()
            if not ret:
                break  # End of video or error

            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings_in_frame:
                # Calculate face distance (lower is more similar)
                face_distance = face_recognition.face_distance([self.reference_encoding], face_encoding)[0]
                if face_distance < self.threshold:
                    frame_count_matching += 1

        video_capture.release()
        cv2.destroyAllWindows()

        return is_match

    def run(self):
        """
        Orchestrates the entire pipeline:
          1) Extract audio,
          2) Transcribe it,
          3) Detect face similarity.
        Returns a dictionary with transcription text and average similarity score.
        """
        # 1) Extract audio
        self.extract_audio()

        # 2) Transcribe audio
        transcription = self.transcribe_audio()
        # print(f"Transcribed text:\n{transcription}")
        detected_digits = find_three_spoken_digits(transcription)
        # 3) Face similarity
        similarity = self.detect_face_similarity()
        return {"3-digit": detected_digits, "similarity": similarity}


def main():
    parser = argparse.ArgumentParser(description="Media Processing Script")

    # Add arguments
    parser.add_argument("--ffmpeg_path", type=str, default="C:/ffmpeg/bin/ffmpeg.exe", help="Path to ffmpeg executable. Default assumes ffmpeg is in PATH")
    parser.add_argument("--video_path", type=str, help="Path to the input video file",
                        default="E:/kuro/test/sequential_prediction/data/RakeshTest video/video-133813067716352727.webm")
    parser.add_argument("--reference_image_path", type=str, help="Path to the reference image file",
                        default="E:/kuro/test/sequential_prediction/data/RakeshTest video/download.jpg")
    parser.add_argument("--output_audio", type=str, default="output_audio.wav", help="Path to store the extracted audio. Default: output_audio.wav")
    parser.add_argument("--threshold", type=float, default=0.65, help="Threshold for face distance (lower means stricter matching). Default: 0.65")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate (Hz) for the extracted audio. Default: 16000")
    parser.add_argument("--video_ratio", type=float, default=0.5, help="Ratio of video to process (0.7 - 1.0). Default: 1.0")

    # Parse arguments
    args = parser.parse_args()

    # Instantiate the MediaProcessor with parsed args
    processor = MediaProcessor(
        video_path=args.video_path,
        reference_image_path=args.reference_image_path,
        output_audio_path=args.output_audio,
        threshold=args.threshold,
        sample_rate=args.sample_rate,
        ffmpeg_path=args.ffmpeg_path,
    )

    # Run the pipeline
    results = processor.run()

    # Print results
    print("\n=== Final Results ===")
    print(results)


if __name__ == "__main__":
    main()
