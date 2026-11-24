import argparse
import json
import os
import sys

from PIL import Image

MODEL_NAME = os.environ.get("IMAGE_MODEL_NAME", "dima806/deepfake_vs_real_image_detection")
DEFAULT_THRESHOLD = float(os.environ.get("IMAGE_FAKE_THRESHOLD", "0.5"))
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_FAKE_WORDS = ("fake", "synthetic", "generated", "artificial", "aiart", "ai_", "ai-", "spoof")
_REAL_WORDS = ("real", "human", "authentic", "genuine", "natural")

_cache = {}


def _load():
    if "model" in _cache:
        return _cache["processor"], _cache["model"], _cache["fake_idx"], _cache["torch"]

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    fake_idx = None
    for idx, name in model.config.id2label.items():
        low = str(name).lower()
        if any(w in low for w in _FAKE_WORDS) and not any(w in low for w in _REAL_WORDS):
            fake_idx = int(idx)
    if fake_idx is None:
        raise RuntimeError("Could not tell which label means FAKE. Labels: %s" % dict(model.config.id2label))

    _cache.update(processor=processor, model=model, fake_idx=fake_idx, torch=torch)
    return processor, model, fake_idx, torch


def score_images(images):
    """List of PIL images -> list of FAKE probabilities (0.0 to 1.0)."""
    processor, model, fake_idx, torch = _load()
    inputs = processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[:, fake_idx]
    return [float(p) for p in probs]


def make_result(fake_prob, threshold=DEFAULT_THRESHOLD):
    is_fake = fake_prob >= threshold
    return {
        "label": "FAKE" if is_fake else "REAL",
        "confidence": round((fake_prob if is_fake else 1.0 - fake_prob) * 100, 2),
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round((1.0 - fake_prob) * 100, 2),
        "threshold": threshold,
    }


def predict_image(image, threshold=DEFAULT_THRESHOLD):
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    return make_result(score_images([image])[0], threshold)


def _main():
    ap = argparse.ArgumentParser(description="Real vs Fake image detector")
    ap.add_argument("--file", help="path to one image")
    ap.add_argument("--folder", help="path to a folder of images")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.file and not args.folder:
        ap.error("give --file or --folder")

    paths = []
    if args.file:
        paths.append(args.file)
    if args.folder:
        for name in sorted(os.listdir(args.folder)):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(args.folder, name))
    if not paths:
        print("No images found.")
        return 1

    results = []
    for p in paths:
        try:
            r = predict_image(p, args.threshold)
        except Exception as exc:
            print("%s -> ERROR: %s" % (p, exc))
            continue
        r["file"] = p
        results.append(r)
        if not args.json:
            print("%s -> %s (%.1f%% confidence | fake %.1f%% / real %.1f%%)" % (
                os.path.basename(p), r["label"], r["confidence"], r["fake_probability"], r["real_probability"]))

    if args.json:
        print(json.dumps(results, indent=2))
    elif len(results) > 1:
        fakes = sum(1 for r in results if r["label"] == "FAKE")
        print("\nSummary: %d images, %d FAKE, %d REAL" % (len(results), fakes, len(results) - fakes))
    return 0


if __name__ == "__main__":
    sys.exit(_main())