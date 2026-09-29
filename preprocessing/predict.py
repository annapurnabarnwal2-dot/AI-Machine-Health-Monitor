import librosa
import numpy as np
import pandas as pd

# -----------------------------
# File paths
# -----------------------------
audio_path = "dataset/fan/id_00/abnormal/normal/00000018.wav"
csv_path = "dataset/fan_features.csv"


# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv(csv_path)

X = data.drop("label", axis=1).values
y = data["label"].values


# -----------------------------
# Load test audio
# -----------------------------
audio, sample_rate = librosa.load(
    audio_path,
    sr=None
)


# -----------------------------
# Extract MFCC
# -----------------------------
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

new_features = np.mean(mfcc, axis=1)


# -----------------------------
# Calculate distance
# -----------------------------
distances = np.sqrt(
    np.sum((X - new_features) ** 2, axis=1)
)


# -----------------------------
# Select nearest 5 samples
# -----------------------------
k = 5

nearest_indices = np.argsort(distances)[:k]

nearest_labels = y[nearest_indices]


# -----------------------------
# Majority voting
# -----------------------------
normal_count = np.sum(nearest_labels == "normal")
abnormal_count = np.sum(nearest_labels == "abnormal")

if normal_count > abnormal_count:
    prediction = "normal"
else:
    prediction = "abnormal"


# -----------------------------
# Display result
# -----------------------------
print("\n================================")
print("      MACHINE HEALTH MONITOR")
print("================================")

print("Audio File:", audio_path)
print("Sample Rate:", sample_rate)
print("Nearest Samples:", k)

print("\nPrediction:", prediction.upper())

if prediction == "normal":
    print("Status: MACHINE NORMAL")
else:
    print("Status: POSSIBLE ABNORMAL CONDITION")

print("================================")
