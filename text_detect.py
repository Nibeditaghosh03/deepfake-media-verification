"""
AI Text Detector
----------------
Detects whether text is REAL (human-written) or FAKE (AI-generated).

Model: fakespot-ai/roberta-base-ai-text-detection-v1
"""

from transformers import pipeline

# ============================================================
# AI TEXT DETECTION MODEL
# ============================================================

print("Loading text detection model...")

classifier = pipeline(
    "text-classification",
    model="fakespot-ai/roberta-base-ai-text-detection-v1"
)

print("Model loaded.\n")


# ============================================================
# TEXT DETECTION FUNCTION
# ============================================================

def detect_text(text: str):
    """
    Runs AI-vs-human detection on a piece of text.
    Returns a dict: {"label": "REAL" or "FAKE", "confidence": float}
    """
    text = text.strip()
    if not text:
        raise ValueError("No text provided.")

    result = classifier(text)[0]

    raw_label = result["label"].upper()
    score = result["score"]

    # Normalize whatever label names the model uses into REAL / FAKE
    if "FAKE" in raw_label or "AI" in raw_label or "MACHINE" in raw_label or "GENERATED" in raw_label:
        label = "FAKE"
    else:
        label = "REAL"

    return {
        "label": label,
        "confidence": score * 100
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":
    real_text = "This is something I genuinely wrote myself about my day."
    fake_text = "The intricate interplay of technological advancement and societal adaptation continues to shape our modern discourse."

    print("REAL sample:", detect_text(real_text))
    print("FAKE sample:", detect_text(fake_text))

    print("\n--- Interactive mode (type 'exit' to quit) ---")
    while True:
        text = input("Enter some text: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue
        result = detect_text(text)
        print(f"\nDetection result:")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.2f}%\n")