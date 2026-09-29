import librosa

# Actual audio file path
audio_path = "dataset/fan/id_00/abnormal/normal/00000018.wav"

# Load audio
audio, sample_rate = librosa.load(audio_path, sr=None)

print("Audio loaded successfully!")
print("Sample Rate:", sample_rate)
print("Number of Samples:", len(audio))

# Calculate duration
duration = len(audio) / sample_rate

print("Duration:", duration, "seconds")
