
# Voice Analysis Application

The Voice Analysis Application developed is designed to analyze and compare the voice frequencies of two audio files to determine how similar the voices are. The application performs the following key functions:

Audio Upload and Frequency Analysis: Users upload two audio files, which are processed to extract frequency data. The system leverages the Web Audio API to decode the audio data and perform frequency analysis using an AnalyserNode. It converts the audio signal into a frequency spectrum that shows the magnitude of different frequency bins.

ONNX.js Model Integration: The application uses ONNX.js to load a pre-trained machine learning model. This model is designed to analyze the voice patterns of both audio files. Once the frequency data is extracted from each file, the model compares the data to determine the similarity between the voices.

Voice Matching and Percentage Calculation: The application then compares the frequency data of both audio files. If the voices are similar, the model calculates and displays a matching percentage that indicates how closely the two voices resemble each other. If the voices differ, it shows that the voice frequencies do not match. This matching percentage is stored in a useState variable (matchPercentage) in React to dynamically update the user interface.

Visual Frequency Data Representation: To give users a better understanding of the analysis, the application uses the react-chartjs-2 library to visualize the frequency data of both audio files. It plots a line chart where each dataset represents the frequency spectrum of an audio file. This chart provides a visual comparison of the voice frequencies from both files.

User Feedback: The results of the analysis, including the duration of each audio file and the matching percentage between the two voices, are displayed in a clear and user-friendly format. The system provides immediate feedback, showing whether the voices match and to what extent.

In summary, this application leverages machine learning and frequency analysis to compare voices, providing a matching percentage to users based on the similarities in audio frequencies.

## Installation - Command
1. First install Python in your system
    * pip install tf2onnx
    * pip install tensorflow
    * pip install librosa
2. npx create-react-app@latest voice-analysis-application --template typescript
3. npm install onnxruntime-web

## Process
1. Create folder "python-script" in src folder and inside "python-script" folder create audio-python-trained-model.py
2. Go to "python-script" folder and open cmd and run script "python audio-python-trained-model.py"
3. Then run again in cmd "python -m tf2onnx.convert --saved-model audio_model.h5 --output audio_model.onnx", it will convert into trained machine learning model in ONNX file.
4. trained model file is used in model path. Example:

5. TSX file: 

import React, { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";
import { Line } from "react-chartjs-2";

interface AudioAnalyzerProps {
  files: File[];
}

const AudioAnalyzer: React.FC<AudioAnalyzerProps> = ({ files }) => {
  const [frequencyData, setFrequencyData] = useState<number[][]>([]);
  const [matchPercentage, setMatchPercentage] = useState<number | null>(null);

  useEffect(() => {
    const analyzeAudios = async () => {
      const newFrequencyData: number[][] = [];

      // Load your custom ONNX model
      const model = await ort.InferenceSession.create("/path/to/audio_model.onnx");

      for (const file of files) {
        const arrayBuffer = await file.arrayBuffer();
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const analyser = audioContext.createAnalyser();
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(analyser);
        analyser.connect(audioContext.destination);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(dataArray);

        newFrequencyData.push(Array.from(dataArray));

        // Process the audio data and extract features using the ONNX model
        const features = await extractVoiceFeatures(model, dataArray);
        console.log("Extracted Features:", features);
      }

      setFrequencyData(newFrequencyData);

      if (newFrequencyData.length === 2) {
        const similarity = calculateVoiceMatch(newFrequencyData[0], newFrequencyData[1]);
        setMatchPercentage(similarity);
      }
    };

    if (files.length > 0) {
      analyzeAudios();
    }
  }, [files]);

  const extractVoiceFeatures = async (model: ort.InferenceSession, frequencyData: number[]) => {
    const inputTensor = new ort.Tensor("float32", Float32Array.from(frequencyData), [1, frequencyData.length]);
    const results = await model.run({ input: inputTensor });
    const output = results["output"]; // Adjust this to your model's output
    return output.data;
  };

  const calculateVoiceMatch = (data1: number[], data2: number[]): number => {
    const minLength = Math.min(data1.length, data2.length);
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < minLength; i++) {
      dotProduct += data1[i] * data2[i];
      magnitudeA += Math.pow(data1[i], 2);
      magnitudeB += Math.pow(data2[i], 2);
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    const similarity = dotProduct / (magnitudeA * magnitudeB);
    return Math.max(0, Math.min(100, similarity * 100)); // Convert to percentage
  };

  return (
    <div>
      <h2>Audio Analysis Results</h2>
      {matchPercentage !== null && <p>Voice Frequency Match: {matchPercentage.toFixed(2)}%</p>}
    </div>
  );
};

export default AudioAnalyzer;



6. Python Script: 

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


## Thankyou ##

