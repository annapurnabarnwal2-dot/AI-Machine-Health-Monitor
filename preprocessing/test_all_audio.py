import os
import librosa
import numpy as np

dataset_path = "dataset/fan/id_00"

features = []
labels = []
files = []


# --------------------------------
# Extract features from all WAV files
# --------------------------------
for root, dirs, filenames in os.walk(dataset_path):

    for filename in filenames:

        if filename.lower().endswith(".wav"):

            file_path = os.path.join(root, filename)

            folder = os.path.basename(root).lower()

            if folder not in ["normal", "abnormal"]:
                continue

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

                feature = np.mean(mfcc, axis=1)

                features.append(feature)
                labels.append(folder)
                files.append(file_path)

            except Exception as e:
                print("Error:", file_path)
                print(e)


X = np.array(features)
y = np.array(labels)


print("\n================================")
print("   MACHINE HEALTH TESTING")
print("================================")

print("Total audio files:", len(X))
print("Normal files:", np.sum(y == "normal"))
print("Abnormal files:", np.sum(y == "abnormal"))


# --------------------------------
# Leave-One-Out KNN testing
# --------------------------------

correct = 0
total = len(X)

for i in range(total):

    test_sample = X[i]

    # All samples except current sample
    train_X = np.delete(X, i, axis=0)
    train_y = np.delete(y, i)

    # Calculate distances
    distances = np.sqrt(
        np.sum((train_X - test_sample) ** 2, axis=1)
    )

    # Find 5 nearest samples
    k = 5

    nearest_indices = np.argsort(distances)[:k]

    nearest_labels = train_y[nearest_indices]

    normal_count = np.sum(
        nearest_labels == "normal"
    )

    abnormal_count = np.sum(
        nearest_labels == "abnormal"
    )

    if normal_count > abnormal_count:
        prediction = "normal"
    else:
        prediction = "abnormal"

    actual = y[i]

    if prediction == actual:
        correct += 1

    print(
        f"{i+1:02d}. "
        f"Actual: {actual:8s} | "
        f"Prediction: {prediction:8s}"
    )


# --------------------------------
# Accuracy
# --------------------------------

accuracy = (correct / total) * 100

print("\n================================")
print("           FINAL RESULT")
print("================================")

print("Total files :", total)
print("Correct     :", correct)
print("Incorrect   :", total - correct)
print(f"Accuracy    : {accuracy:.2f}%")

print("================================")
