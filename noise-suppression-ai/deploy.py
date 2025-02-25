import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import time

# Load trained model
model = tf.keras.models.load_model("noise_suppression_model.h5")

# Function to process real-time audio input
def preprocess_audio(audio, sample_rate=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)

# Function to capture and classify real-time audio
def real_time_noise_detection(duration=2, sample_rate=22050):
    print("Listening for noise...")
    while True:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        processed_audio = preprocess_audio(audio.flatten(), sample_rate)
        prediction = model.predict(processed_audio)[0][0]
        
        if prediction > 0.5:
            print("🔊 Noise detected! Applying suppression...")
        else:
            print("✅ Clean audio detected.")

        time.sleep(1)

# Start real-time detection
real_time_noise_detection()
