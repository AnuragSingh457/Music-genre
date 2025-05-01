import tensorflow as tf
keras = tf.keras
models = tf.keras.models
import streamlit as st
import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import to_categorical
import pickle

# Load pre-trained model and label encoder
model = load_model('music_genre_classifier.h5')  # Replace with your model's filename

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Preprocessing function for incoming audio
MAX_PAD_LEN = 174
N_MFCC = 40

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load full audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    # Normalize per sample
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Pad or truncate to fixed length
    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    return mfcc

# Streamlit UI
st.title('Music Genre Classifier')

st.write("""
    Upload an audio file (WAV format) and our model will classify the music genre!
""")

# File uploader
audio_file = st.file_uploader("Choose an audio file...", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    # Save audio to a temporary file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Extract features from the uploaded audio file
    features = extract_features("temp_audio.wav")
    features = features[np.newaxis, ..., np.newaxis]  # Add batch and channel dimension
    
    # Make a prediction
    prediction = model.predict(features)
    genre_index = np.argmax(prediction)
    genre = le.inverse_transform([genre_index])[0]

    st.write(f"Predicted Genre: **{genre}**")
