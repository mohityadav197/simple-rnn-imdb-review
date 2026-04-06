import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# 1. Load the IMDB word index
word_index = imdb.get_word_index()

# 2. Load the Bidirectional LSTM Model
model = load_model('lstm_model.h5')

# 3. FIXED Helper Function: Preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    # The Keras IMDB standard: 
    # 0=Padding, 1=Start, 2=OOV (Unknown), 3=Unused. 
    # Real words start at index 4 (original index + 3).
    encoded_review = []
    for word in words:
        index = word_index.get(word, None)
        # If word is missing or outside our 10,000 vocab, use the OOV token (2)
        if index is None or index >= 10000:
            encoded_review.append(2)
        else:
            encoded_review.append(index + 3)
            
    # Ensure input is exactly 250 integers long
    padded_review = sequence.pad_sequences([encoded_review], maxlen=250)
    return padded_review

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Sentiment Pro V2.1", page_icon="🎬")

# Sidebar: Project Roadmap & Versioning
st.sidebar.title("🚀 Project Roadmap")
st.sidebar.success("**Current Version:** V2.1 (The 'Logic Fix')")
st.sidebar.write("""
✅ **Preprocessing Patch:** Fixed the OOV indexing bug. Now the model correctly identifies 'logicless' and other rare negative words.
""")

st.sidebar.info("""
📊 **Performance Stage:**
- **Architecture:** Bidirectional LSTM
- **Vocab Size:** 10,000 words
- **Fix:** Corrected word-to-index mapping.
""")

st.sidebar.write("---")
st.sidebar.write(f"Developed by Mohit | AI Engineering Portfolio 2026")

# Main Content
st.title("🎬 AI Movie Review Sentiment Analysis")
st.write("V2.1: Now with corrected input processing for higher accuracy.")

user_input = st.text_area("Enter your movie review here:", "The movie was not good at all and the story was logicless.")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.error("Please enter a review first!")
    else:
        with st.spinner('Analyzing deep context...'):
            # Preprocess and Predict
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            
            # Binary classification threshold
            sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
            confidence = prediction[0][0] if sentiment == "Positive" else 1 - prediction[0][0]

            # Display Result
            col1, col2 = st.columns(2)
            
            with col1:
                if sentiment == "Positive":
                    st.success(f"### Sentiment: {sentiment} ✅")
                else:
                    st.error(f"### Sentiment: {sentiment} ❌")
            
            with col2:
                st.metric("Confidence Score", f"{confidence * 100:.2f}%")

            # Technical Breakdown
            with st.expander("What changed in V2.1?"):
                st.write("""
                We fixed the 'Indexing Handshake'. Rare words are now correctly passed to the LSTM 
                as 'Unknown' tokens rather than being misidentified as common words.
                """)
                st.write(f"**Model Raw Output:** {prediction[0][0]:.4f}")

else:
    st.write("Click analyze to see the V2.1 logic in action.")