from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import torch
import librosa

MODEL = "DavidCombei/wavLM-base-Deepfake_V2"
AUDIO_FILE = "i-want-your-money.wav"

print("Loading processor...")

processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)

print("Loading model...")

model = AutoModelForAudioClassification.from_pretrained(MODEL)

model.eval()

print("Model loaded successfully!")

print("\nAnalyzing:", AUDIO_FILE)

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=16000,
    mono=True
)

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)

print("\nRESULT")
print("=" * 60)

print("Model labels:")
print(model.config.id2label)

print("Label mapping:")
print("LABEL_0 =", model.config.id2label[0])
print("LABEL_1 =", model.config.id2label[1])

print("\nProbabilities:")

for i, probability in enumerate(probabilities[0]):
    label = model.config.id2label[i]
    percentage = probability.item() * 100

    print(f"{label}: {percentage:.2f}%")

predicted_index = torch.argmax(probabilities[0]).item()
predicted_label = model.config.id2label[predicted_index]

print("\nPredicted:", predicted_label)

print("=" * 60)