import streamlit as st
import numpy as np
import tensorflow as tf
import re # NEW: For punctuation cleaning
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# 1. Load the IMDB word index
word_index = imdb.get_word_index()

# 2. Load the Bidirectional LSTM Model
model = load_model('lstm_model.h5')

# 3. THE ULTIMATE PREPROCESSING PATCH
def preprocess_text(text):
    # Clean text: remove punctuation and lowercase everything
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    # EVERY review must start with index 1 (The 'Start Signal')
    encoded_review = [1] 
    
    for word in words:
        index = word_index.get(word, None)
        # Handle Out of Vocabulary (OOV)
        if index is None or index >= 10000:
            encoded_review.append(2) # 2 is the 'Unknown' token
        else:
            encoded_review.append(index + 3) # IMDB standard shift
            
    # Final Padding to match the 250 length from training
    padded_review = sequence.pad_sequences([encoded_review], maxlen=250)
    return padded_review

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Sentiment Pro V2.2", page_icon="🎬")

st.sidebar.title("🚀 Project Roadmap")
st.sidebar.success("**Current Version:** V2.2 (The Final Handshake)")
st.sidebar.write("✅ **Fixed:** Punctuation handling & Start Tokens.")

st.title("🎬 AI Movie Review Sentiment Analysis")
st.write("V2.2: Corrected logic for high-accuracy negation handling.")

user_input = st.text_area("Enter your review:", "This movie is bad. I hated it.")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        with st.spinner('Analyzing context...'):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            
            sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
            confidence = prediction[0][0] if sentiment == "Positive" else 1 - prediction[0][0]

            col1, col2 = st.columns(2)
            with col1:
                if sentiment == "Positive":
                    st.success(f"### Sentiment: {sentiment} ✅")
                else:
                    st.error(f"### Sentiment: {sentiment} ❌")
            with col2:
                st.metric("Confidence Score", f"{confidence * 100:.2f}%")