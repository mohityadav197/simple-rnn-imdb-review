import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import h5py
import json

# --- 1. ROBUST MODEL LOADER (The "Metadata Patch") ---
def safe_load_model(model_path):
    try:
        return load_model(model_path)
    except (TypeError, ValueError, AttributeError):
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                raw_config = f.attrs['model_config']
                # Handle both bytes and string types for 2026 environments
                if isinstance(raw_config, bytes):
                    raw_config = raw_config.decode('utf-8')
                
                model_config = json.loads(raw_config)
                
                for layer in model_config['config']['layers']:
                    if 'config' in layer:
                        # Strip Keras 3 specific keywords
                        layer['config'].pop('quantization_config', None)
                        if 'batch_shape' in layer['config']:
                            layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                
                f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
        return load_model(model_path)

# --- 2. APP CONFIGURATION & DATA ---
st.set_page_config(page_title="RNN Sentiment Pro", page_icon="🎬", layout="wide")

# Load word index once
@st.cache_resource
def get_imdb_data():
    return imdb.get_word_index(), safe_load_model('simple_rnn_imdb.h5')

word_index, model = get_imdb_data()

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=250)
    return padded_review

# --- 3. UI LAYOUT: SIDEBAR ---
with st.sidebar:
    st.image("https://www.tensorflow.org/images/tf_logo_social.png", width=150)
    st.title("Neural Network Info")
    st.markdown("""
    **Architecture:** Simple RNN  
    **Dataset:** IMDB Reviews  
    **Input Shape:** $(None, 250)$  
    **Vocabulary:** 10,000 words
    """)
    st.divider()
    st.write("Built at 5:00 AM in Kapriwas 🇮🇳")

# --- 4. UI LAYOUT: MAIN PANEL ---
st.title("🎬 AI Movie Review Sentiment Analysis")
st.write("This system uses a Recurrent Neural Network to predict if a review is positive or negative.")

user_review = st.text_area("Enter your movie review here:", height=150, 
                          placeholder="Example: The acting was superb but the plot was a bit slow...")

if st.button("Analyze Sentiment"):
    if user_review.strip():
        with st.spinner('Neural Network is processing...'):
            # Preprocessing
            processed_input = preprocess_text(user_review)
            
            # Prediction Logic
            prediction = model.predict(processed_input)
            prob = float(prediction[0][0])
            
            # Logic: If $P(positive) > 0.5$, sentiment is Positive
            sentiment = "Positive" if prob > 0.5 else "Negative"
            confidence = prob if sentiment == "Positive" else 1 - prob

            st.divider()
            
            # Metric Columns
            col1, col2 = st.columns(2)
            with col1:
                if sentiment == "Positive":
                    st.success(f"### Sentiment: {sentiment} ✅")
                else:
                    st.error(f"### Sentiment: {sentiment} ❌")
            
            with col2:
                st.metric(label="Confidence Score", value=f"{confidence:.2%}")
                st.progress(prob)

            # Debugging/Interactivity expander
            with st.expander("Technical Breakdown"):
                st.write("The model processed your text into the following numerical sequence:")
                st.code(processed_input)
                st.write(f"Raw Output Probability: {prob:.4f}")
    else:
        st.warning("Please enter some text before analyzing.")

st.markdown("---")
st.caption("Deployment Status: Dockerized | CI/CD: GitHub Actions (Live)")