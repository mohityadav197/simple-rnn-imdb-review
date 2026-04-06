🎬 AI Sentiment Pro V2.2: The Bidirectional LSTM Journey
"It's not just about the model; it's about the data handshake."
🚀 Project Overview
This project is a Deep Learning-based sentiment analysis tool that classifies movie reviews as Positive or Negative. While many beginner projects stop at a simple RNN, this version represents a full evolution through three architectural iterations to solve real-world logical failures like negation, context-switching, and data misalignment.

📈 The Evolution (Honest Report)
V1.0: The Simple RNN (Structural Failure)
Architecture: Basic Recurrent Neural Network.

The Reality: The model suffered from the Vanishing Gradient Problem. It could not "remember" words from the beginning of a sentence by the time it reached the end.

The Fail Case: It classified "The movie was not good" as Positive because it effectively "forgot" the word "not."

V2.0: Bidirectional LSTM (The Brain Upgrade)
Architecture: Dual-layer Long Short-Term Memory (LSTM).

The Reality: Added "Hindsight." The model now reads the sentence forward and backward simultaneously.

The Discovery: Even with a better architecture, the app still struggled. This led to a deep-dive investigation into the "Indexing Handshake Bug."

V2.2: The Final Handshake (Production Grade)
The Fix: I identified a critical mismatch between the Keras IMDB training dataset and the production inference script.

Key Correction: Implemented a +3 index shift, a mandatory <START> token (Index 1), and a robust <OOV> (Index 2) handler to ensure the model "reads" exactly what it was "taught."

🔬 Honest Performance Analysis
Case 1: The "High-Evidence" Negative
Review: "This movie was bad. I hated it."

Result: 99.48% Negative

Analysis: When provided with strong, clear negative features, the Bidirectional LSTM reaches near-total mathematical certainty.

Case 2: The "Diluted" Negation
Review: "This movie was not good."

Result: 56.90% Negative

Analysis: The model correctly identified the negation ("not"), but the confidence is lower because the short 4-word "signal" is mathematically diluted by the 246 zero-padding tokens required for the input window.

Case 3: The "Sarcasm" Limit (Feature Dominance)
Review: "A masterpiece, but a total letdown."

Result: 88.15% Positive

The Reality: High-weight words like "masterpiece" have massive statistical gravity in the IMDB dataset. In this case, the positive weight of "masterpiece" outcompeted the negative weight of "letdown," proving that LSTMs still struggle with complex sarcasm where "Feature Dominance" overrides "Syntactic Logic."

🛠️ Tech Stack
Framework: TensorFlow / Keras

Model: Bidirectional LSTM

Deployment: Streamlit Cloud

Pre-processing: Regex Sanitization & IMDB Index Mapping

Dataset: IMDB (10,000 Vocab, 250 Sequence Length)

🧠 Lessons for the 2026 IT Market
Sanitization is Mandatory: A perfect model will fail if punctuation (like the period in "bad.") isn't stripped, as it turns key words into "Unknown" tokens.

The Start Signal: Every model expects a "handshake." Without the <START> token, the model's internal sequence is shifted, leading to "hallucinated" results.

Statistical vs. Logical: Deep learning models are statistical engines. Understanding why a model fails (like Case 3) is more valuable than pretending it is 100% accurate.