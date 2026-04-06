import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# 1. Load the IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# 2. Load the NEW Bidirectional LSTM Model (V2.0)
# Ensure 'lstm_model.h5' is the filename you saved in your notebook
model = load_model('lstm_model.h5')

# 3. Helper Function: Preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    # Map words to integers; words not in vocab become 2 (OOV)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Ensure the input is exactly 250 numbers long
    padded_review = sequence.pad_sequences([encoded_review], maxlen=250)
    return padded_review

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Sentiment Pro V2.0", page_icon="🎬")

# Sidebar: Project Roadmap & Versioning
st.sidebar.title("🚀 Project Roadmap")
st.sidebar.success("**Current Version:** V2.0 (Bidirectional LSTM)")
st.sidebar.write("""
✅ **Upgraded Brain:** The model now reads reviews **forward and backward** simultaneously to understand deep context and negation (like "not good").
""")

st.sidebar.info("""
📊 **Performance Stage:**
- **Architecture:** Bidirectional LSTM
- **Vocab Size:** 10,000 words
- **Status:** Production Ready
""")

st.sidebar.write("---")
st.sidebar.write(f"Developed by Mohit | AI Engineering Portfolio 2026")

# Main Content
st.title("🎬 AI Movie Review Sentiment Analysis")
st.write("Using Deep Learning (LSTM) to understand the true emotion behind a review.")

user_input = st.text_area("Enter your movie review here:", "The movie was not good at all and the story was logicless.")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.error("Please enter a review first!")
    else:
        with st.spinner('Bidirectional LSTM is analyzing context...'):
            # Preprocess and Predict
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
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
            with st.expander("Why is V2.0 better?"):
                st.write("""
                Unlike the Simple RNN (V1.0), this **LSTM** uses a 'Cell State' to remember words from the beginning of the sentence. 
                Because it is **Bidirectional**, it catches 'negation' (like 'not') regardless of where it appears.
                """)
                st.write(f"**Model Raw Output:** {prediction[0][0]:.4f}")

else:
    st.write("Click the button above to test the V2.0 Intelligence.")