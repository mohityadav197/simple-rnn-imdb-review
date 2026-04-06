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

# 3. CORRECTED PREPROCESSING (MATCHES TRAINING DATA EXACTLY)
def preprocess_text(text):
    """
    Preprocess text to match training data format from imdb.load_data().
    
    IMPORTANT: imdb.load_data() returns sequences WITHOUT a [1] start token.
    The [1] is metadata, not part of review sequences.
    
    Index mapping (after imdb.load_data()):
    - 0: padding
    - 1: start of sequence (NOT prepended in actual data)
    - 2: unknown/OOV token
    - 3+: word indices (word_index value + 3)
    """
    # Step 1: Clean text - remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    # Step 2: Encode words WITHOUT prepending [1]
    # Matches structure of imdb.load_data() training sequences
    encoded_review = []
    
    for word in words:
        index = word_index.get(word, None)
        # Handle Out of Vocabulary (OOV)
        if index is None or index >= 10000:
            # Use 2 for unknown tokens (reserved in IMDB encoding)
            encoded_review.append(2)
        else:
            # Add 3 to match IMDB's internal encoding scheme
            encoded_review.append(index + 3)
    
    # Step 3: Pad to 250 length (matches training padding)
    # pad_sequences pads with 0 by default (which is correct)
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