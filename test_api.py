import requests

AUDIO_FILE = "i-want-your-money.wav"

ENDPOINT_URL = "PASTE_YOUR_ENDPOINT_URL_HERE"
HF_TOKEN = "PASTE_YOUR_HF_TOKEN_HERE"


headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "audio/wav"
}


with open(AUDIO_FILE, "rb") as audio:
    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        data=audio
    )


print("=" * 60)
print("HUGGING FACE API RESULT")
print("=" * 60)

print("Status code:", response.status_code)
print("Response:")

try:
    print(response.json())
except Exception:
    print(response.text)

print("=" * 60)