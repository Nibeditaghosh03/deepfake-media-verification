---

title: Deepfake Media Verification
emoji: ⚡
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Real vs fake detection for voice, text, image and live video
---

# Deepfake Media Detector: Voice + Text + Image + Video

A web dashboard and command-line toolset that checks whether media is **real or AI-generated**. It covers four kinds of media:

- **Voice audio**: human or AI-generated speech
- **Text**: human-written or AI-generated writing
- **Images**: real or AI-generated / manipulated
- **Video**: real or deepfake, including a **live** camera mode (`/live`)

It is built on pretrained models, so no training is required and no API keys are needed.

## What's inside

| File | Purpose |
|---|---|
| `app.py` | Flask backend: serves the dashboard and the detection APIs |
| `detect.py` | Voice detection logic (CLI + used by `app.py`) |
| `text_detect.py` | Text detection logic (CLI + used by `app.py`) |
| `image_detect.py` | Image detection logic |
| `video_detect.py` | Video detection logic (uploaded files) |
| `video_live.py` | Live video detection logic |
| `media_routes.py` | API routes for image and video uploads |
| `live_routes.py` | API routes for the live camera page |
| `templates/index.html` | Main dashboard (Voice, Text, Image and Video tabs) |
| `templates/live.html` | Live camera page (`/live`) |
| `static/style.css` | Dashboard styling |
| `static/script.js` | Mic recording, file upload, tab switching, calls to the backend |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages (`ffmpeg`, `libsndfile1`, `libglib2.0-0`) for hosted deployment |
| `test_api.py`, `hf_audio_test.py`, `hf_pipeline_test.py` | Small test scripts |

## Models used

- **Voice:** `DavidCombei/wavLM-base-Deepfake_V2` (WavLM-based classifier): LABEL_0 = FAKE, LABEL_1 = REAL
- **Text:** `fakespot-ai/roberta-base-ai-text-detection-v1` (RoBERTa-based classifier)
- **Image:** `<IMAGE MODEL NAME>`
- **Video:** `<VIDEO MODEL NAME>` (frames are extracted with OpenCV, and faces are found with an OpenCV Haar cascade)

The models download automatically from Hugging Face the first time you run the app, then are cached locally for every run after that.

## 1. Setup (one time)

Requires Python 3 and Git.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## 2. Run the dashboard

```powershell
python app.py
```

Wait for:

```
Running on http://127.0.0.1:7860
```

Open that link in your browser.

### Voice tab
- **Start Recording**, speak (or play an AI voice clip near your mic), then **Stop Recording**. The board shows REAL or FAKE with a confidence score.
- Or use **Upload File** to check an existing `.wav`, `.mp3`, `.flac`, `.ogg` or `.m4a` clip.

### Text tab
- Paste or type text and click **Check Text**. The board shows REAL (human-written) or FAKE (AI-generated) with a confidence score.

### Image tab
- Upload an image (`.jpg`, `.png`, `.webp`). The board shows REAL or FAKE with a confidence score.

### Video tab
- Upload a short video clip. The app samples frames, runs the detector on them, and shows REAL or FAKE with a confidence score.

### Live mode
- Open `http://127.0.0.1:7860/live` and allow camera access. The page checks the camera feed continuously and updates the result as it runs.

## 3. Or use the command line directly

**Voice:**
```powershell
python detect.py --file myclip.wav
python detect.py --folder my_clips/
python detect.py --record 5          # record 5s from mic, check once
python detect.py --live 5            # keep recording+checking every 5s until Ctrl+C
```

**Text:**
```powershell
python text_detect.py                # runs built-in samples, then interactive mode
```

## How testing works

- Speaking normally into your own mic will (correctly) show REAL every time.
- To test the FAKE side of voice detection, play an AI-generated voice clip (from any TTS or voice-cloning tool) out loud near your mic while recording, or upload a saved AI-generated audio file.
- To test the FAKE side of text detection, paste AI-generated writing from any LLM.
- To test images and video, use one real file and one AI-generated or manipulated file of each kind, and compare the results.

## Deployment notes

- **Run locally:** the app works fully on a normal laptop (steps above).
- **Hosted demo:** the models and the PyTorch libraries are large, so free hosting plans with small memory limits may not run the app. A hosted copy needs enough RAM to hold two or more models.
- **Public link for a demo:** while the app runs on your laptop, a tunnel tool can give it a temporary public HTTPS link. HTTPS is also needed for browser mic and camera access on any remote link.

## Known limitations

- No detector is 100% accurate. Treat results as a strong signal, not proof.
- **Voice:** works best on clear speech clips 3+ seconds long with minimal background noise. Very short or noisy clips are less reliable, and newer or higher-quality voice cloning tools can sometimes evade detection.
- **Text:** works best on 50+ words of English text. The model may be less accurate on writing from newer LLMs it wasn't specifically trained to catch.
- **Image and video:** compression, low resolution, heavy filters, or faces that are small or turned away can reduce accuracy. Video detection depends on finding a clear face in the frames.
- **Live mode:** results depend on lighting, camera quality and computer speed. On a slow CPU the live check updates slowly.