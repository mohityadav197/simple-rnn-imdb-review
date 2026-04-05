🎬 RNN Sentiment Pro: Production-Ready AI Pipeline
An end-to-end Natural Language Processing (NLP) project featuring a Recurrent Neural Network (RNN) deployed via a fully automated CI/CD pipeline and Docker.

🚀 Overview
This project classifies movie reviews as Positive or Negative using an RNN model trained on the IMDB dataset. Beyond the AI logic, it demonstrates professional-grade software engineering practices, including cross-version model compatibility and containerized deployment.

🛠️ Tech Stack
AI/ML: TensorFlow, Keras, NumPy

Web Framework: Streamlit

DevOps: Docker, GitHub Actions (CI/CD)

Infrastructure: Docker Hub

✨ Key Features
Real-time Sentiment Analysis: Predictive modeling for custom text input.

Pro-Level UI: A "wide-mode" dashboard featuring confidence metrics, progress bars, and a technical breakdown sidebar.

Environment-Agnostic Loader: A custom "Safe Load" utility that patches Keras 3 metadata for Keras 2 environments, preventing deployment crashes.

Automated Pipeline: Fully integrated CI/CD that builds and pushes the image to Docker Hub on every commit.

🤖 CI/CD Pipeline
This repository uses GitHub Actions to maintain a "Build-Once, Run-Anywhere" workflow:

Code Push: Triggered on every push to the main branch.

Containerization: GitHub spins up a Linux runner to build the Docker image.

Automated Registry: The image is pushed to Docker Hub (mohitkhairwal2005/simple-rnn-imdb-review).

📦 Getting Started
1. Run via Docker (Recommended)
You don't need Python or TensorFlow installed. Just pull the latest production image:

Bash
# Pull the image from Docker Hub
docker pull mohitkhairwal2005/simple-rnn-imdb-review:latest

# Run the container
docker run -p 8501:8501 mohitkhairwal2005/simple-rnn-imdb-review:latest
Access the app at: http://localhost:8501

2. Local Development
Bash
git clone https://github.com/mohityadav197/simple-rnn-imdb-review.git
cd simple-rnn-imdb-review
pip install -r requirements.txt
streamlit run main.py
📂 Project Structure
Plaintext
├── .github/workflows/
│   └── deploy.yml          # CI/CD Pipeline configuration
├── main.py                 # Streamlit UI with Metadata Patching logic
├── simple_rnn_imdb.h5      # Pre-trained RNN Model
├── Dockerfile              # Container configuration
├── requirements.txt        # Managed dependencies (TensorFlow 2.16+)
└── README.md               # Documentation
🧠 Engineering Highlights for Recruiters
Cross-Version Compatibility: Implemented a robust h5py metadata stripper to allow Keras 3 models to run in Keras 2 environments.

Security: Managed deployment secrets using GitHub Secrets to prevent API/Token leakage.

Optimization: Utilized @st.cache_resource for efficient model loading and resource management.

Maintained by: Mohit Yadav | Location: Kapriwas, Haryana 🇮🇳