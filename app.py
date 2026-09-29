from flask import Flask, render_template, request
import librosa
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

CSV_PATH = "dataset/fan_features.csv"


def predict_audio(audio_path):

    # Load dataset
    data = pd.read_csv(CSV_PATH)

    X = data.drop("label", axis=1).values
    y = data["label"].values

    # Load audio
    audio, sample_rate = librosa.load(
        audio_path,
        sr=None
    )

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    # Convert MFCC into 13 features
    new_features = np.mean(mfcc, axis=1)

    # Calculate distance from dataset
    distances = np.sqrt(
        np.sum((X - new_features) ** 2, axis=1)
    )

    # Select 5 nearest samples
    k = 5

    nearest_indices = np.argsort(distances)[:k]

    nearest_labels = y[nearest_indices]

    # Count votes
    normal_count = np.sum(
        nearest_labels == "normal"
    )

    abnormal_count = np.sum(
        nearest_labels == "abnormal"
    )

    # AI prediction
    if normal_count > abnormal_count:
        prediction = "normal"
    else:
        prediction = "abnormal"

    # Simple vote-based confidence
    confidence = (
        max(normal_count, abnormal_count) / k
    ) * 100

    return (
        prediction,
        sample_rate,
        normal_count,
        abnormal_count,
        confidence,
        new_features
    )


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    sample_rate = None
    normal_votes = None
    abnormal_votes = None
    confidence = None
    filename = None
    mfcc_features = None
    error = None

    if request.method == "POST":

        if "audio" not in request.files:

            error = "Please select an audio file."

        else:

            file = request.files["audio"]

            if file.filename == "":

                error = "Please select an audio file."

            elif not file.filename.lower().endswith(".wav"):

                error = "Only WAV files are supported."

            else:

                filename = file.filename

                os.makedirs(
                    "uploads",
                    exist_ok=True
                )

                file_path = os.path.join(
                    "uploads",
                    filename
                )

                file.save(file_path)

                try:

                    (
                        result,
                        sample_rate,
                        normal_votes,
                        abnormal_votes,
                        confidence,
                        mfcc_features
                    ) = predict_audio(file_path)

                except Exception as e:

                    error = str(e)

    return render_template(
        "index.html",
        result=result,
        sample_rate=sample_rate,
        normal_votes=normal_votes,
        abnormal_votes=abnormal_votes,
        confidence=confidence,
        filename=filename,
        mfcc_features=mfcc_features,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)