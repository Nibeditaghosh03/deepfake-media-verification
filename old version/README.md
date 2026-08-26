# Audio Fake vs Real Detector

Detects whether a speech clip is **REAL** (human) or **FAKE** (AI-generated / deepfake / voice-cloned).
Runs on a real pretrained model (`garystafford/wav2vec2-deepfake-voice-detector`, a fine-tuned Wav2Vec2 classifier) — no training required.

## 1. Install (one time)

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

That's it. No API keys needed.

## 2. Run detection

Check a single file:
```bash
python detect.py --file sample.wav
```

Check every audio file in a folder:
```bash
python detect.py --folder my_clips/
```

Record from your mic and check it once (5 seconds in this example):
```bash
python detect.py --record 5
```

Continuous live detection — records and checks 5-second chunks back-to-back until you press `Ctrl+C`:
```bash
python detect.py --live 5
```

Supported formats: `.wav .mp3 .flac .ogg .m4a`

## Example output

```
File: sample.wav
  Verdict:    🔴 FAKE
  Confidence: 93.42%
  Breakdown:  REAL: 6.58%, FAKE: 93.42%
```

## 3. Visual "board" (recommended)

Instead of the terminal, you can use a simple web dashboard: a big REAL/FAKE indicator with a Record button and an Upload button.

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

- Click **Start Recording**, speak, click **Stop Recording** → the board turns green (REAL) or red (FAKE) with a confidence score.
- Or click **Upload File** to check an existing audio clip the same way.

To actually test the FAKE side: play an AI-generated voice clip (from any TTS/voice-cloning tool) into your mic while recording, or upload a saved AI-generated audio file. Speaking normally into your own mic will (correctly) show REAL every time.


- First run downloads the model (~1-2 GB) from Hugging Face and caches it locally — after that, detection is instant offline.
- Uses GPU automatically if you have CUDA available, otherwise runs on CPU (a bit slower but works fine).
- Best accuracy on clear speech clips, 3+ seconds long, minimal background noise.
- No detector is 100% accurate — treat the confidence score as a strong signal, not absolute proof, especially on very short or noisy clips.
