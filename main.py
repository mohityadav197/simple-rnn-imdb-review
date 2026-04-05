import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import h5py
import json

# --- THE "SAFE LOAD" FIX ---
# This function strips Keras 3 metadata that Keras 2 doesn't understand
def safe_load_model(model_path):
    try:
        # Step 1: Try loading normally
        return load_model(model_path)
    except (TypeError, ValueError) as e:
        st.warning("Detected Keras version mismatch. Patching model metadata...")
        
        # Step 2: Manually strip the offending 'quantization_config' and 'batch_shape'
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                
                # Dig into the layers to remove Keras 3 specific keys
                for layer in model_config['config']['layers']:
                    if 'config' in layer:
                        # Remove 'quantization_config' if it exists
                        layer['config'].pop('quantization_config', None)
                        # Remove 'batch_shape' if it exists (Keras 2 uses 'batch_input_shape')
                        if 'batch_shape' in layer['config']:
                            layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                
                f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
        
        return load_model(model_path)

# --- APP LOGIC ---

# 1. Load the Word Index (Required for preprocessing)
word_index = imdb.get_word_index()

# 2. Load the Model using our Safe Loader
model = safe_load_model('simple_rnn_imdb.h5')

# 3. Helper Function to Preprocess Text
def preprocess_text(text):
    words = text.lower().split()
    # Map words to indices, using 2 for unknown words (standard IMDB practice)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Padding to match the model's expected input length (250)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=250)
    return padded_review

# 4. Streamlit UI
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Type a review below to see if the RNN thinks it is Positive or Negative.')

user_input = st.text_area('Movie Review', 'This movie was fantastic! I loved the acting.')

if st.button('Classify'):
    # Preprocess the user input
    preprocessed_input = preprocess_text(user_input)
    
    # Make Prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display Result
    st.subheader(f'Sentiment: {sentiment}')
    st.write(f'Confidence Score: {prediction[0][0]:.4f}')
else:
    st.write('Please enter a review.')