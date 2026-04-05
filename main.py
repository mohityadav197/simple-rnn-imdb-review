import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import h5py
import json

# --- THE ROBUST SAFE LOAD FIX ---
def safe_load_model(model_path):
    try:
        # Attempt 1: Standard Load
        return load_model(model_path)
    except (TypeError, ValueError, AttributeError) as e:
        st.warning("Patching model metadata for environment compatibility...")
        
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                # The Fix: Handle both 'bytes' and 'str' types for metadata
                raw_config = f.attrs['model_config']
                if isinstance(raw_config, bytes):
                    raw_config = raw_config.decode('utf-8')
                
                model_config = json.loads(raw_config)
                
                # Iterate through layers to strip Keras 3 specific keywords
                for layer in model_config['config']['layers']:
                    if 'config' in layer:
                        # Remove Keras 3 quantization tags
                        layer['config'].pop('quantization_config', None)
                        # Map Keras 3 batch_shape back to Keras 2 batch_input_shape
                        if 'batch_shape' in layer['config']:
                            layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                
                # Save the "cleaned" metadata back to the file
                f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
        
        # Attempt 2: Load the patched model
        return load_model(model_path)

# --- APP INITIALIZATION ---

# 1. Load the Word Index
# Note: This might take a second the first time as it downloads 1.6MB
word_index = imdb.get_word_index()

# 2. Load the Model using our Robust Loader
model = safe_load_model('simple_rnn_imdb.h5')

# 3. Text Preprocessing Logic
def preprocess_text(text):
    words = text.lower().split()
    # The IMDB index is offset by 3 (0=padding, 1=start, 2=unknown)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Ensure it's the exact length the RNN expects (250)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=250)
    return padded_review

# --- STREAMLIT UI ---
st.title('🎬 RNN Movie Sentiment Analyzer')
st.write("Deploying deep learning models to production at 4:30 AM.")

user_review = st.text_area('Enter your movie review here:', 'I really enjoyed this film, the story was very compelling!')

if st.button('Analyze Sentiment'):
    if user_review.strip():
        # Process and Predict
        processed_input = preprocess_text(user_review)
        prediction = model.predict(processed_input)
        
        # Interpret Result
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        confidence = prediction[0][0] if sentiment == 'Positive' else 1 - prediction[0][0]
        
        # Display results with some style
        st.subheader(f'The model thinks this is: {sentiment}')
        st.progress(float(prediction[0][0]))
        st.write(f"Confidence Score: {confidence:.2%}")
    else:
        st.error("Please enter some text to analyze.")

st.sidebar.info("Model: Simple RNN | Dataset: IMDB | Deployment: Docker + GitHub Actions")