import os
import librosa
import numpy as np
import pandas as pd

dataset_path = "dataset/fan/id_00"

features = []
labels = []

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.lower().endswith(".wav"):

            file_path = os.path.join(root, file)

            try:
                audio, sample_rate = librosa.load(
                    file_path,
                    sr=None
                )

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sample_rate,
                    n_mfcc=13
                )

                mfcc_features = np.mean(mfcc, axis=1)

                # Label based on the immediate folder
                folder_name = os.path.basename(root).lower()

                if folder_name == "normal":
                    label = "normal"
                elif folder_name == "abnormal":
                    label = "abnormal"
                else:
                    print("Skipped unknown folder:", file_path)
                    continue

                features.append(mfcc_features)
                labels.append(label)

                print("Processed:", file_path)
                print("Label:", label)

            except Exception as e:
                print("Error:", file_path)
                print(e)


# Create DataFrame
columns = [f"MFCC_{i+1}" for i in range(13)]

df = pd.DataFrame(features, columns=columns)

df["label"] = labels

# Save CSV
df.to_csv(
    "dataset/fan_features.csv",
    index=False
)

print("\n==============================")
print("Dataset created successfully!")
print("==============================")
print("Total files:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nCSV saved: dataset/fan_features.csv")