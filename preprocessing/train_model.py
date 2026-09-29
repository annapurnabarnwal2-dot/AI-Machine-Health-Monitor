import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load dataset
data = pd.read_csv("dataset/fan_features.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Features and labels
X = data.drop("label", axis=1)
y = data["label"]

print("\nClass distribution:")
print(y.value_counts())


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Prediction
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model
joblib.dump(model, "fan_health_model.pkl")

print("\nModel saved as: fan_health_model.pkl")
