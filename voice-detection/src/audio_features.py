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

def extract_chroma(audio, sample_rate):
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate
    )

    chroma_mean = np.mean(chroma, axis=1)

    return chroma_mean
def extract_zero_crossing_rate(audio):
    zcr = librosa.feature.zero_crossing_rate(audio)

    zcr_mean = np.mean(zcr)

    return zcr_mean

def extract_spectral_centroid(audio, sample_rate):
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    centroid_mean = np.mean(centroid)

    return centroid_mean

def extract_spectral_rolloff(audio, sample_rate):
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate
    )

    rolloff_mean = np.mean(rolloff)

    return rolloff_mean

def extract_rms_energy(audio):
    rms = librosa.feature.rms(y=audio)

    rms_mean = np.mean(rms)

    return rms_mean

def extract_features(file_path):
    audio, sample_rate = load_audio(file_path)

    mfcc = extract_mfcc(audio, sample_rate)
    chroma = extract_chroma(audio, sample_rate)
    zcr = extract_zero_crossing_rate(audio)
    centroid = extract_spectral_centroid(audio, sample_rate)
    rolloff = extract_spectral_rolloff(audio, sample_rate)
    rms = extract_rms_energy(audio)

    features = np.concatenate([
        mfcc,
        chroma,
        [zcr],
        [centroid],
        [rolloff],
        [rms]
    ])

    return features