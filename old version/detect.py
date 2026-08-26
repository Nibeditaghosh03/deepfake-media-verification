"""
AI Voice Deepfake Detector
--------------------------

Detects whether speech audio is:

    REAL  = human speech
    FAKE  = AI-generated / synthetic / voice-cloned speech

Model:
    garystafford/wav2vec2-deepfake-voice-detector

The model's documented classes are:
    Class 0 -> REAL
    Class 1 -> FAKE

Usage:

    python detect.py --file audio.wav

    python detect.py --file audio.mp3

    python detect.py --folder my_audio_folder

    python detect.py --record 5

    python detect.py --live 5
"""

import argparse
import os
import sys
import tempfile

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

SAMPLE_RATE = 16000

# The model documentation says 2.5-13 seconds is the
# optimal training/input range.
CHUNK_SECONDS = 5

# Minimum useful audio length.
MIN_AUDIO_SECONDS = 0.5

# Probability threshold.
# >= 0.50 -> FAKE
# <  0.50 -> REAL
FAKE_THRESHOLD = 0.50

SUPPORTED_EXTS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
)

# Global model objects
_model = None
_feature_extractor = None
_device = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the pretrained Wav2Vec2 deepfake detector.

    The model is downloaded from Hugging Face on first run
    and then cached locally.
    """

    global _model
    global _feature_extractor
    global _device

    if _model is not None:
        return

    import torch
    from transformers import (
        AutoModelForAudioClassification,
        AutoFeatureExtractor,
    )

    print()
    print("=" * 70)
    print("Loading AI voice deepfake detection model")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print()

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {_device}")

    if _device == "cuda":
        print("GPU acceleration enabled.")
    else:
        print("Running on CPU.")
        print("Inference may take longer.")

    print()
    print("Loading processor...")

    _feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    _model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME
    )

    _model.to(_device)
    _model.eval()

    print()
    print("Model loaded successfully.")
    print()

    # IMPORTANT:
    # The model documentation specifies:
    #
    # Class 0 = REAL
    # Class 1 = FAKE
    #
    # Print the actual config too so we can diagnose problems.

    print("Model configuration:")
    print(f"  Number of classes: {_model.config.num_labels}")
    print(f"  id2label: {_model.config.id2label}")
    print()

    if _model.config.num_labels != 2:
        print("WARNING:")
        print("Expected a binary classifier with 2 classes.")
        print()

    print("=" * 70)
    print()


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_path):
    """
    Load audio as mono 16 kHz floating-point waveform.
    """

    import librosa

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print(f"Loading audio: {audio_path}")

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio is None or len(audio) == 0:
        raise ValueError("Audio file contains no usable audio.")

    audio = np.asarray(audio, dtype=np.float32)

    duration = len(audio) / SAMPLE_RATE

    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {duration:.2f} seconds")

    if duration < MIN_AUDIO_SECONDS:
        print()
        print("WARNING:")
        print(
            f"Audio is shorter than "
            f"{MIN_AUDIO_SECONDS} seconds."
        )
        print("Prediction may be unreliable.")
        print()

    return audio


# ============================================================
# CREATE AUDIO CHUNKS
# ============================================================

def create_chunks(audio):
    """
    Split audio into approximately 5-second chunks.

    This is useful because the model was trained on relatively
    short speech clips rather than arbitrary-length recordings.
    """

    chunk_size = SAMPLE_RATE * CHUNK_SECONDS

    chunks = []

    total_samples = len(audio)

    for start in range(0, total_samples, chunk_size):

        end = start + chunk_size

        chunk = audio[start:end]

        if len(chunk) == 0:
            continue

        # Ignore extremely tiny final fragments.
        if len(chunk) < SAMPLE_RATE * MIN_AUDIO_SECONDS:
            continue

        # If the final chunk is shorter than 5 seconds,
        # pad it with zeros.
        if len(chunk) < chunk_size:

            padding = chunk_size - len(chunk)

            chunk = np.pad(
                chunk,
                (0, padding),
                mode="constant"
            )

        chunks.append(chunk.astype(np.float32))

    return chunks


# ============================================================
# PREDICT ONE CHUNK
# ============================================================

def predict_chunk(audio_chunk):
    """
    Run the model on one audio chunk.

    Returns:

        real_probability
        fake_probability
    """

    import torch

    inputs = _feature_extractor(
        audio_chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(_device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = _model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # IMPORTANT:
    #
    # Official model mapping:
    #
    # index 0 = REAL
    # index 1 = FAKE

    real_probability = float(probabilities[0])
    fake_probability = float(probabilities[1])

    return real_probability, fake_probability


# ============================================================
# PREDICT COMPLETE AUDIO
# ============================================================

def predict(audio_path):
    """
    Analyze an audio file.

    The audio is divided into chunks.

    Every chunk is classified separately.

    Final prediction is based on the average FAKE probability.

    Returns:

        label
        confidence
        details
    """

    load_model()

    audio = load_audio(audio_path)

    chunks = create_chunks(audio)

    if not chunks:
        raise ValueError(
            "Could not create usable audio chunks."
        )

    print()
    print(f"Analyzing {len(chunks)} audio chunk(s)...")
    print()

    results = []

    for index, chunk in enumerate(chunks, start=1):

        real_probability, fake_probability = predict_chunk(
            chunk
        )

        results.append({
            "chunk": index,
            "real": real_probability,
            "fake": fake_probability
        })

        chunk_label = (
            "FAKE"
            if fake_probability >= FAKE_THRESHOLD
            else "REAL"
        )

        print(
            f"Chunk {index:02d}: "
            f"{chunk_label:5s} | "
            f"REAL = {real_probability * 100:6.2f}% | "
            f"FAKE = {fake_probability * 100:6.2f}%"
        )

    # ========================================================
    # AGGREGATE ALL CHUNKS
    # ========================================================

    fake_probabilities = np.array(
        [item["fake"] for item in results],
        dtype=np.float32
    )

    real_probabilities = np.array(
        [item["real"] for item in results],
        dtype=np.float32
    )

    average_fake = float(
        np.mean(fake_probabilities)
    )

    average_real = float(
        np.mean(real_probabilities)
    )

    # Also calculate median.
    #
    # Median helps reduce the influence of one unusual chunk.

    median_fake = float(
        np.median(fake_probabilities)
    )

    median_real = float(
        np.median(real_probabilities)
    )

    # Final decision:
    #
    # Average probability is the main decision.
    #
    # >= 50% fake -> FAKE
    # <  50% fake -> REAL

    if average_fake >= FAKE_THRESHOLD:

        label = "FAKE"

        confidence = average_fake

    else:

        label = "REAL"

        confidence = average_real

    details = {
        "average_real": average_real,
        "average_fake": average_fake,
        "median_real": median_real,
        "median_fake": median_fake,
        "chunks": results,
    }

    return label, confidence * 100, details


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    path,
    label,
    confidence,
    details
):
    """
    Print a clean final prediction.
    """

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"File:       {path}")

    if label == "FAKE":

        print("Verdict:    🔴 FAKE")

    else:

        print("Verdict:    🟢 REAL")

    print(f"Confidence: {confidence:.2f}%")

    print()
    print("Probability breakdown:")
    print(
        f"  REAL: {details['average_real'] * 100:.2f}%"
    )
    print(
        f"  FAKE: {details['average_fake'] * 100:.2f}%"
    )

    print()
    print("Median probabilities:")
    print(
        f"  REAL: {details['median_real'] * 100:.2f}%"
    )
    print(
        f"  FAKE: {details['median_fake'] * 100:.2f}%"
    )

    print()
    print(
        f"Chunks analyzed: "
        f"{len(details['chunks'])}"
    )

    print("=" * 70)
    print()


# ============================================================
# RECORD AUDIO
# ============================================================

def record_audio(seconds):
    """
    Record audio from the default microphone.
    """

    import sounddevice as sd
    import soundfile as sf

    if seconds <= 0:
        raise ValueError(
            "Recording duration must be greater than 0."
        )

    print()
    print("=" * 70)
    print(f"Recording {seconds} seconds...")
    print("Speak now.")
    print("=" * 70)

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print()
    print("Recording complete.")

    tmp_path = os.path.join(
        tempfile.gettempdir(),
        "deepfake_detector_recording.wav"
    )

    sf.write(
        tmp_path,
        audio,
        SAMPLE_RATE
    )

    print(f"Saved temporary recording: {tmp_path}")

    return tmp_path


# ============================================================
# LIVE DETECTION
# ============================================================

def live_loop(seconds):
    """
    Continuously record and analyze audio.
    """

    load_model()

    print()
    print("=" * 70)
    print(
        f"LIVE DETECTION: "
        f"{seconds}-second chunks"
    )
    print("Press Ctrl+C to stop.")
    print("=" * 70)
    print()

    round_number = 1

    try:

        while True:

            print()
            print(
                f"================ ROUND "
                f"{round_number} ================"
            )

            path = record_audio(seconds)

            try:

                label, confidence, details = predict(
                    path
                )

                print_result(
                    path,
                    label,
                    confidence,
                    details
                )

            except Exception as error:

                print()
                print(
                    f"Error analyzing recording: "
                    f"{error}"
                )

            round_number += 1

    except KeyboardInterrupt:

        print()
        print("Live detection stopped.")


# ============================================================
# SCAN FOLDER
# ============================================================

def scan_folder(folder_path):
    """
    Analyze every supported audio file in a folder.
    """

    if not os.path.isdir(folder_path):

        print(
            f"Error: folder not found: "
            f"{folder_path}"
        )

        sys.exit(1)

    files = []

    for filename in sorted(
        os.listdir(folder_path)
    ):

        full_path = os.path.join(
            folder_path,
            filename
        )

        if (
            os.path.isfile(full_path)
            and filename.lower().endswith(
                SUPPORTED_EXTS
            )
        ):

            files.append(full_path)

    if not files:

        print(
            "No supported audio files found."
        )

        print(
            "Supported formats:"
        )

        print(
            ", ".join(SUPPORTED_EXTS)
        )

        sys.exit(1)

    print()
    print(
        f"Found {len(files)} audio file(s)."
    )
    print()

    for index, file_path in enumerate(
        files,
        start=1
    ):

        print()
        print(
            f"############ FILE "
            f"{index}/{len(files)} ############"
        )

        try:

            label, confidence, details = predict(
                file_path
            )

            print_result(
                file_path,
                label,
                confidence,
                details
            )

        except Exception as error:

            print(
                f"Error processing "
                f"{file_path}:"
            )

            print(error)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Detect whether speech audio is "
            "REAL or AI-generated FAKE."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--file",
        type=str,
        help="Path to an audio file."
    )

    group.add_argument(
        "--folder",
        type=str,
        help="Path to a folder containing audio files."
    )

    group.add_argument(
        "--record",
        type=int,
        metavar="SECONDS",
        help="Record audio from microphone."
    )

    group.add_argument(
        "--live",
        type=int,
        metavar="SECONDS",
        help="Continuously record and analyze audio."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    if args.live is not None:

        live_loop(args.live)

        return

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    if args.record is not None:

        path = record_audio(
            args.record
        )

        try:

            label, confidence, details = predict(
                path
            )

            print_result(
                path,
                label,
                confidence,
                details
            )

        except Exception as error:

            print()
            print(
                f"Detection error: {error}"
            )

            sys.exit(1)

        return

    # --------------------------------------------------------
    # FILE
    # --------------------------------------------------------

    if args.file is not None:

        if not os.path.isfile(args.file):

            print(
                f"Error: file not found: "
                f"{args.file}"
            )

            sys.exit(1)

        try:

            label, confidence, details = predict(
                args.file
            )

            print_result(
                args.file,
                label,
                confidence,
                details
            )

        except Exception as error:

            print()
            print(
                "=" * 70
            )

            print(
                "DETECTION ERROR"
            )

            print(
                "=" * 70
            )

            print(error)

            print(
                "=" * 70
            )

            sys.exit(1)

        return

    # --------------------------------------------------------
    # FOLDER
    # --------------------------------------------------------

    if args.folder is not None:

        scan_folder(
            args.folder
        )

        return


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()