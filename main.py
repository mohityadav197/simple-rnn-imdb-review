import io
import os
import re
import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# --- Configuration ---
MAX_LEN = 250
VOCAB_SIZE = 10000
MODEL_FILE = "simple_rnn_imdb.h5"

# --- UI Styling ---
def inject_styles():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        }
        .stButton>button {
            background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
            color: white;
            border: none;
            box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3);
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(139, 92, 246, 0.5);
            filter: brightness(1.1);
        }
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>div>div {
            background: rgba(30, 41, 59, 0.8) !important;
            color: #f1f5f9 !important;
            border: 2px solid rgba(139, 92, 246, 0.4) !important;
            border-radius: 10px !important;
        }
        .stMetric {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
            border: 2px solid rgba(139, 92, 246, 0.3);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .stExpander {
            background: rgba(30, 41, 59, 0.5);
            border: 2px solid rgba(139, 92, 246, 0.2);
            border-radius: 10px;
        }
        .header-box {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%);
            padding: 30px;
            border-radius: 20px;
            border: 2px solid rgba(139, 92, 246, 0.4);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            margin-bottom: 30px;
        }
        .sentiment-positive {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.2) 100%);
            border-left: 5px solid #22c55e;
            padding: 15px;
            border-radius: 10px;
        }
        .sentiment-negative {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
            border-left: 5px solid #ef4444;
            padding: 15px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def cached_resource(func):
    try:
        return st.cache_resource(func)
    except AttributeError:
        return st.cache(func)


def cached_data(func):
    try:
        return st.cache_data(func)
    except AttributeError:
        return st.cache(func)


# --- Model & Data Loading ---
@cached_resource
def load_model_and_vocabulary():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Expected model file '{MODEL_FILE}' not found.")

    model = load_model(MODEL_FILE, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return model, word_index, reverse_word_index


@cached_data
def load_test_data(sample_size: int = 2000):
    (_, _), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)
    return X_test[:sample_size], y_test[:sample_size]

# --- Processing Logic ---
def clean_review_text(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [token for token in text.split() if token]

def encode_review(tokens: list[str], word_index: dict) -> np.ndarray:
    encoded = []
    for token in tokens:
        index = word_index.get(token, None)
        # IMDB indexing: 0=pad, 1=start, 2=oov. Actual word indices start at 3.
        if index is None or index >= VOCAB_SIZE:
            encoded.append(2)
        else:
            encoded.append(index + 3)
    if not encoded:
        encoded = [2]
    return sequence.pad_sequences([encoded], maxlen=MAX_LEN)

def decode_ids(encoded_review: list[int], reverse_word_index: dict) -> str:
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review if i >= 3])

def predict_sentiment(review: str, model, word_index: dict) -> tuple[str, float, np.ndarray]:
    tokens = clean_review_text(review)
    encoded = encode_review(tokens, word_index)
    prediction = model.predict(encoded, verbose=0)
    score = float(np.nan_to_num(prediction[0][0], nan=0.0))
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score, encoded

def render_insights(tokens: list[str], word_index: dict, encoded: np.ndarray, reverse_word_index: dict):
    total = len(tokens)
    known = sum(1 for t in tokens if t in word_index)
    coverage = (known / total) if total > 0 else 0

    st.markdown("### Review Intelligence")
    c1, c2, c3 = st.columns(3)
    c1.metric("Token Count", total)
    c2.metric("Known Tokens", known)
    c3.metric("Vocab Coverage", f"{coverage:.0%}")

    if total > 0:
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        top_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        st.write("**Top Tokens Found:** " + ", ".join([f"{k} ({v})" for k, v in top_tokens]))

# --- Main App Interface ---
def main():
    st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="wide")
    inject_styles()

    st.markdown(
        "<div class='header-box'>"
        "<h1 style='margin:0; font-size: 3em; background: linear-gradient(135deg, #8b5cf6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🎬 IMDB Sentiment Analyzer</h1>"
        "<p style='color:#cbd5e1; margin-top:10px; font-size:1.1em;'>Advanced Deep Learning RNN Model for Movie Review Sentiment Analysis</p>"
        "</div>",
        unsafe_allow_html=True
    )

    try:
        model, word_index, reverse_word_index = load_model_and_vocabulary()
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return

    # Tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Analysis", "📊 Batch Analysis", "⚙️ Model Settings", "📈 Statistics"])

    with tab1:
        col_input, col_config = st.columns([3, 1], gap="medium")
        
        with col_input:
            st.subheader("Input & Analysis")
            input_mode = st.radio("📝 Input Method", ["Custom Review", "Sample Review"], horizontal=True)
            
            sample_reviews = [
                "🟢 This was an absolute masterpiece! The acting was incredible and the story kept me engaged.",
                "🔴 I found the plot confusing and the characters quite dull. A waste of time.",
                "🟡 A decent movie for a one-time watch, but forgettable in the long run.",
                "🟢 Outstanding cinematography and brilliant performances all around!",
                "🔴 Completely disappointed. Poor screenplay and boring narrative."
            ]
            
            if input_mode == "Sample Review":
                review_text = st.selectbox("Choose a sample", sample_reviews, format_func=lambda x: x[5:])
            else:
                review_text = st.text_area("Enter your movie review", "The story was engaging and the actors delivered...", height=120)
        
        with col_config:
            st.subheader("⚙️ Config")
            threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            show_details = st.checkbox("Show Details", True)
        
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            analyze_button = st.button("🚀 Analyze Now", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🔄 Clear", use_container_width=True)
        with col_btn3:
            history_button = st.button("📜 Show History", use_container_width=True)

        if analyze_button:
            if review_text.strip():
                sentiment, score, encoded = predict_sentiment(review_text, model, word_index)
                tokens = clean_review_text(review_text)
                
                # Results Display
                col_sentiment, col_score, col_confidence = st.columns(3)
                
                with col_sentiment:
                    if score >= threshold:
                        sentiment_color = "green"
                        emoji = "😊"
                        label = "POSITIVE"
                    else:
                        sentiment_color = "red"
                        emoji = "😞"
                        label = "NEGATIVE"
                    
                    st.markdown(
                        f"<div style='text-align:center; padding:20px; background:rgba({255 if sentiment_color=='red' else 34},{150 if sentiment_color=='green' else 68},68,0.1); "
                        f"border-radius:10px; border-left:5px solid {('red' if sentiment_color=='red' else 'green')};'>"
                        f"<h3>{emoji} {label}</h3>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                with col_score:
                    st.metric("Score", f"{score:.4f}", f"{abs(score - 0.5) * 2:.2%} confidence")
                
                with col_confidence:
                    st.metric("Threshold", f"{threshold:.2f}", "Boundary value")
                
                # Progress Bar
                progress_color = "#22c55e" if score >= threshold else "#ef4444"
                st.markdown(
                    f"<div style='height:10px; background:rgba(139,92,246,0.2); border-radius:10px; overflow:hidden;'>"
                    f"<div style='height:100%; width:{score*100}%; background:linear-gradient(90deg, #8b5cf6, {progress_color});'></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                if show_details:
                    with st.expander("📊 Detailed Analysis", expanded=True):
                        render_insights(tokens, word_index, encoded, reverse_word_index)
                    
                    with st.expander("🏗️ Model Architecture"):
                        buf = io.StringIO()
                        model.summary(print_fn=lambda x: buf.write(x + "\n"))
                        st.code(buf.getvalue(), language="text")
                
                # Save to session state for history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    'review': review_text[:100],
                    'sentiment': sentiment,
                    'score': score
                })
            else:
                st.warning("Please enter a review first!")

    with tab2:
        st.subheader("📊 Batch Review Analysis")
        batch_text = st.text_area("Paste multiple reviews (one per line)", height=200)
        
        if st.button("Analyze Batch"):
            reviews = [r.strip() for r in batch_text.split('\n') if r.strip()]
            if reviews:
                with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, rev in enumerate(reviews):
                        sent, score, _ = predict_sentiment(rev, model, word_index)
                        results.append({'Review': rev[:80], 'Sentiment': sent, 'Score': f'{score:.3f}'})
                        progress_bar.progress((idx + 1) / len(reviews))
                    
                    st.dataframe(results, use_container_width=True)
            else:
                st.warning("Please enter at least one review!")

    with tab3:
        st.subheader("⚙️ Model Configuration")
        
        col_diag1, col_diag2 = st.columns(2)
        
        with col_diag1:
            if st.button("🧪 Run Diagnostics", use_container_width=True):
                with st.spinner("Evaluating on test dataset..."):
                    X_test, y_test = load_test_data(sample_size=500)
                    loss, acc = model.evaluate(X_test, y_test, verbose=0)
                    
                    col_acc, col_loss = st.columns(2)
                    col_acc.metric("Test Accuracy", f"{acc:.2%}", "Overall Performance")
                    col_loss.metric("Test Loss", f"{loss:.4f}", "Error Rate")
                    st.success("✅ Diagnostics Complete!")
        
        with col_diag2:
            st.write("**Model Information:**")
            st.info(f"✓ Model: SimpleRNN\n✓ Vocabulary Size: {VOCAB_SIZE}\n✓ Max Length: {MAX_LEN}\n✓ File: {MODEL_FILE}")

    with tab4:
        st.subheader("📈 Analysis Statistics")
        
        if 'history' in st.session_state and st.session_state.history:
            history = st.session_state.history
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                positive_count = sum(1 for h in history if h['sentiment'] == 'Positive')
                st.metric("Positive Reviews", positive_count, f"out of {len(history)}")
            
            with col_stats2:
                negative_count = sum(1 for h in history if h['sentiment'] == 'Negative')
                st.metric("Negative Reviews", negative_count, f"out of {len(history)}")
            
            with col_stats3:
                avg_score = np.mean([h['score'] for h in history])
                st.metric("Avg Score", f"{avg_score:.3f}", "Mean sentiment")
            
            st.write("**Analysis History:**")
            for idx, item in enumerate(reversed(history[-10:]), 1):
                emoji = "🟢" if item['sentiment'] == 'Positive' else "🔴"
                st.write(f"{idx}. {emoji} {item['review']}... - Score: {item['score']:.3f}")
        else:
            st.info("No analysis history yet. Analyze some reviews first!")

if __name__ == "__main__":
    main()