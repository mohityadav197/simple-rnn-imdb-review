# IMDB Sentiment Analyzer

A Streamlit web application for sentiment analysis of movie reviews using a pre-trained RNN model.

## Features

- Analyze custom movie reviews for sentiment (positive/negative)
- View prediction confidence and token insights
- Run diagnostics on IMDB test data
- Beautiful dark theme UI

## Prerequisites

- Docker
- Docker Compose (optional, for easier management)

## Running with Docker

### Using Docker Compose (Recommended)

1. Build and run the application:
   ```bash
   docker-compose up --build
   ```

2. Open your browser and go to `http://localhost:8501`

### Using Docker directly

1. Build the Docker image:
   ```bash
   docker build -t imdb-sentiment-analyzer .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 imdb-sentiment-analyzer
   ```

3. Open your browser and go to `http://localhost:8501`

## Local Development

If you want to run locally without Docker:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

## Project Structure

- `main.py`: Main Streamlit application
- `simple_rnn_imdb.h5`: Pre-trained RNN model
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker image configuration
- `docker-compose.yml`: Docker Compose configuration
- Notebooks: Jupyter notebooks for model training and analysis



## docker specific structure
Dockerfile - Defines the container image with Python 3.9, installs dependencies, and runs the Streamlit app
.dockerignore - Excludes unnecessary files from the Docker build context
docker-compose.yml - Provides an easy way to run the containerized app
requirements.txt - Added streamlit dependency
README.md - Complete instructions for running the dockerized application