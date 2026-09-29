import librosa
import numpy as np

audio_path = "dataset/fan/id_00/abnormal/normal/00000018.wav"

# Load audio
audio, sample_rate = librosa.load(audio_path, sr=None)

# Extract MFCC features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

# Take mean of each MFCC coefficient
mfcc_features = np.mean(mfcc, axis=1)

print("Audio loaded successfully!")
print("Sample Rate:", sample_rate)
print("MFCC Shape:", mfcc.shape)
print("MFCC Features:")
print(mfcc_features)