"""
AI Voice Deepfake Detector
--------------------------
REAL = human speech
FAKE = AI-generated / synthetic / voice-cloned speech

Model: DavidCombei/wavLM-base-Deepfake_V2
Label mapping: LABEL_0 = FAKE, LABEL_1 = REAL
"""

import os
import sys
import argparse
import tempfile

import numpy as np

MODEL_NAME = "DavidCombei/wavLM-base-Deepfake_V2"
SAMPLE_RATE = 16000
SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

_model = None
_feature_extractor = None
_device = None


def load_model():
    global _model, _feature_extractor, _device
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

    print(f"Loading model '{MODEL_NAME}'...")
    _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    _model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device)
    _model.eval()
    print(f"Model loaded on device: {_device}")
    print(f"Label mapping: {_model.config.id2label}  (LABEL_0=FAKE, LABEL_1=REAL)\n")


def predict(audio_path: str):
    """
    Runs fake/real detection on a single audio file.
    Returns (label, confidence_percent, prob_dict)
    """
    import torch
    import librosa

    load_model()

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32)

    duration = len(audio) / SAMPLE_RATE
    if duration < 0.5:
        print(f"  Warning: '{audio_path}' is very short ({duration:.2f}s); result may be unreliable.")

    inputs = _feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # LABEL_0 = FAKE, LABEL_1 = REAL for this model
    fake_prob = float(probs[0])
    real_prob = float(probs[1])

    if fake_prob >= real_prob:
        label, confidence = "FAKE", fake_prob
    else:
        label, confidence = "REAL", real_prob

    prob_dict = {"REAL": real_prob, "FAKE": fake_prob}
    return label, confidence * 100, prob_dict


def print_result(path, label, confidence, prob_dict):
    icon = "🟢 REAL" if label == "REAL" else "🔴 FAKE"
    print(f"File: {path}")
    print(f"  Verdict:    {icon}")
    print(f"  Confidence: {confidence:.2f}%")
    breakdown = ", ".join(f"{k}: {v*100:.2f}%" for k, v in prob_dict.items())
    print(f"  Breakdown:  {breakdown}")
    print()


def record_audio(seconds: int) -> str:
    import sounddevice as sd
    import soundfile as sf

    print(f"Recording {seconds} seconds from your microphone... speak now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    print("Recording complete.\n")

    tmp_path = os.path.join(tempfile.gettempdir(), "detector_recording.wav")
    sf.write(tmp_path, audio, SAMPLE_RATE)
    return tmp_path


def live_loop(seconds: int):
    load_model()
    print(f"Live mode: recording {seconds}s chunks continuously. Press Ctrl+C to stop.\n")
    round_num = 1
    try:
        while True:
            print(f"--- Round {round_num} ---")
            path = record_audio(seconds)
            label, confidence, prob_dict = predict(path)
            print_result(path, label, confidence, prob_dict)
            round_num += 1
    except KeyboardInterrupt:
        print("\nStopped live detection.")


def main():
    parser = argparse.ArgumentParser(description="Detect if audio is REAL or AI-generated FAKE speech.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single audio file")
    group.add_argument("--folder", type=str, help="Path to a folder of audio files to scan")
    group.add_argument("--record", type=int, metavar="SECONDS", help="Record N seconds from mic and check it once")
    group.add_argument("--live", type=int, metavar="SECONDS", help="Continuously record+check N-second chunks until Ctrl+C")
    args = parser.parse_args()

    if args.live:
        live_loop(args.live)
        return

    if args.record:
        path = record_audio(args.record)
        label, confidence, prob_dict = predict(path)
        print_result(path, label, confidence, prob_dict)
        return

    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
        label, confidence, prob_dict = predict(args.file)
        print_result(args.file, label, confidence, prob_dict)
        return

    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"Error: folder not found: {args.folder}")
            sys.exit(1)
        files = [
            os.path.join(args.folder, f)
            for f in sorted(os.listdir(args.folder))
            if f.lower().endswith(SUPPORTED_EXTS)
        ]
        if not files:
            print(f"No supported audio files found in {args.folder}")
            sys.exit(1)
        print(f"Scanning {len(files)} file(s)...\n")
        for f in files:
            try:
                label, confidence, prob_dict = predict(f)
                print_result(f, label, confidence, prob_dict)
            except Exception as e:
                print(f"  Error processing {f}: {e}\n")
        return


if __name__ == "__main__":
    main()