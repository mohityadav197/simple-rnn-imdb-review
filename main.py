import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# 1. Load the IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# 2. Load the pre-trained model (Simple RNN V1.0)
# Ensure 'simple_rnn_imdb.h5' is in your root directory on GitHub
model = load_model('simple_rnn_imdb.h5')

# 3. Helper Function: Decode reviews (for technical breakdown)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# 4. Helper Function: Preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=250)
    return padded_review

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Sentiment Pro", page_icon="🎬")

# Sidebar: Project Status & Versioning
st.sidebar.title("🚀 Project Roadmap")
st.sidebar.info("**Current Version:** V1.0 (Simple RNN)")
st.sidebar.warning("""
⚠️ **Known Limitation:** V1.0 uses a Simple RNN architecture. It may struggle with:
- Negation (e.g., "not good")
- Sarcasm
- Long-range dependencies
""")

st.sidebar.write("---")
st.sidebar.success("""
🛠 **Coming Soon (V2.0):**
- Switching to **LSTM** architecture.
- Adding **Dropout** to reduce over-fitting.
- Improved Tokenization.
""")

st.sidebar.write(f"Built by Mohit | Kapriwas, IN")

# Main Content
st.title("🎬 AI Movie Review Sentiment Analysis")
st.write("This system uses a Recurrent Neural Network to predict if a review is positive or negative.")

user_input = st.text_area("Enter your movie review here:", "The movie was not good at all and the story was logicless.")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.error("Please enter a review first!")
    else:
        with st.spinner('AI is thinking...'):
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

            # Technical Breakdown (Expandable)
            with st.expander("See Technical Breakdown"):
                st.write(f"**Model Raw Output:** {prediction[0][0]:.4f}")
                st.write("**Note:** Scores closer to 1.0 are Positive, scores closer to 0.0 are Negative.")
                st.info("Notice: Simple RNNs sometimes focus on key words (like 'good') while ignoring qualifiers (like 'not'). This is why we are moving to LSTM.")

else:
    st.write("Click the button above to see the AI's magic.")