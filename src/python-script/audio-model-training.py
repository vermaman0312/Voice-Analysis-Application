import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split

# Example function to extract MFCCs
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load your dataset of audio files and labels (you should provide these)
audio_files = ['audio1.wav', 'audio2.wav', ...]  # Replace with your files
labels = [0, 1, ...]  # Replace with your labels

# Extract features
X = np.array([extract_mfcc(f) for f in audio_files])
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple model (you can experiment with CNNs, RNNs, etc.)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('audio_model.h5')
