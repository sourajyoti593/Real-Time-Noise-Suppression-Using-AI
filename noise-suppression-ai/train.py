import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Load and preprocess audio data
def load_audio_data(directory, sample_rate=22050):
    X, y = [], []
    for label, class_name in enumerate(["clean", "noisy"]):  # Two classes: Clean and Noisy audio
        class_dir = os.path.join(directory, class_name)
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            signal, sr = librosa.load(file_path, sr=sample_rate)
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)  # Extract MFCC features
            X.append(mfccs)
            y.append(label)
    return np.array(X), np.array(y)

# Load dataset
X, y = load_audio_data("dataset/")
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # Reshape for CNN input

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("noise_suppression_model.h5")

# Plot accuracy
plt.plot(model.history.history['accuracy'], label='Train Accuracy')
plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()
