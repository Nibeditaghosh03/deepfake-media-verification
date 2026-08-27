# Deepfake Media Detector — Voice + Text

A web dashboard and CLI toolset that detects whether **voice audio** is human or AI-generated, and whether **written text** is human-written or AI-generated. Built on real pretrained models — no training required, no API keys needed.

## What's inside

| File | Purpose |
|---|---|
| `app.py` | Flask backend serving the web dashboard and both detection APIs |
| `detect.py` | Voice detection logic (CLI + used by `app.py`) |
| `text_detect.py` | Text detection logic (CLI + used by `app.py`) |
| `templates/index.html` | Dashboard UI — Voice tab + Text tab |
| `static/style.css` | Dashboard styling |
| `static/script.js` | Mic recording, file upload, tab switching, calls to backend |
| `requirements.txt` | Python dependencies |

## Models used

- **Voice:** `DavidCombei/wavLM-base-Deepfake_V2` (WavLM-based classifier) — LABEL_0 = FAKE, LABEL_1 = REAL
- **Text:** `fakespot-ai/roberta-base-ai-text-detection-v1` (RoBERTa-based classifier)

Both download automatically from Hugging Face the first time you run the app, then are cached locally for every run after that.

## 1. Setup (one time)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Run the dashboard

```powershell
python app.py
```

Wait for:
```
Running on http://127.0.0.1:5000
```

Open that link in your browser.

### Voice tab
- **Start Recording** → speak (or play an AI voice clip near your mic) → **Stop Recording** → board shows 🟢 REAL or 🔴 FAKE with confidence
- Or **Upload File** to check an existing `.wav`/`.mp3`/`.flac`/`.ogg`/`.m4a` clip

### Text tab
- Paste or type text → **Check Text** → board shows 🟢 REAL (human-written) or 🔴 FAKE (AI-generated) with confidence

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
- To test the FAKE side of voice detection, play an AI-generated voice clip (from any TTS/voice-cloning tool, or Windows' built-in `System.Speech` synthesizer) out loud near your mic while recording, or upload a saved AI-generated audio file directly.
- To test the FAKE side of text detection, paste AI-generated writing from any LLM (ChatGPT, Claude, Gemini, etc.).

## Known limitations

- No detector is 100% accurate — treat results as a strong signal, not proof.
- Voice: works best on clear speech clips 3+ seconds long with minimal background noise; very short or noisy clips are less reliable; newer/higher-quality voice cloning tools can sometimes evade detection.
- Text: works best on 50+ words of English text; this model may be less accurate on writing from newer LLMs it wasn't specifically trained to catch.
