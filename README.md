# Twitter Sentiment Analysis 📊

A machine learning project that analyzes sentiment in airline-related tweets using NLP.

## Overview
This project classifies tweets as **positive, negative, or neutral** using a trained machine learning model on airline customer feedback data.

## Features
- Text preprocessing & cleaning
- Sentiment classification model
- Prediction interface using Streamlit
- Real dataset training

## Tech Stack
- Python
- Pandas
- Scikit-learn
- NLP
- Streamlit

## Dataset
Airline Twitter Sentiment dataset containing real customer tweets.

## 🚀 Project Evolution (Real-Time → Demo Deployment)

This project was originally designed as a **real-time Twitter sentiment analysis system**.

### Original Implementation
The initial version:
- Pulled live tweets using the Twitter API (Tweepy)
- Cleaned and preprocessed tweet text
- Passed tweets through a trained NLP pipeline
- Predicted sentiment (Positive / Neutral / Negative)
- Stored results in a SQLite database
- Visualized sentiment trends over time

Example live queries included topics such as:
- Pakistan vs India discussions
- airline service feedback
- trending social topics

Due to Twitter API free-tier rate limits and deployment restrictions, the public Streamlit version uses a **stable demo dataset (Airline Tweets Dataset)** for reproducible results.

📄 Full methodology and real-time pipeline details are documented in the project report.

## 📄 Project Report

The full academic implementation, including:
- live tweet extraction pipeline
- preprocessing workflow
- supervised ML training
- evaluation & visualization

can be found here:

👉 [Twitter Sentiment Analysis Report](docs/Twitter_Sentiment_Report.docx)


## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

