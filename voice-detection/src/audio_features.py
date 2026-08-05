import librosa
import numpy as np


def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path, sr=16000)

    return audio, sample_rate
def extract_mfcc(audio, sample_rate, n_mfcc=40):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean