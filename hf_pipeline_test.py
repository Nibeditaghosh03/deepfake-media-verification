from transformers import pipeline

MODEL = "DavidCombei/wavLM-base-DeepFake_V2"

AUDIO_FILE = "real_voice.wav"

print("Loading model...")

classifier = pipeline(
    "audio-classification",
    model=MODEL
)

print("Model loaded.")
print()
print("Analyzing:", AUDIO_FILE)

results = classifier(AUDIO_FILE)

print()
print("RESULT")
print("=" * 60)

for result in results:
    print(result)