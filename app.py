import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

st.title("Twitter Sentiment Analysis ✈️")

df = pd.read_csv("twitter airline training data.csv")
df = df[["text", "airline_sentiment"]]
df.columns = ["tweet", "sentiment"]

pipeline = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("model", LogisticRegression(max_iter=200))
])

pipeline.fit(df["tweet"], df["sentiment"])

keyword = st.text_input("Enter keyword")

if keyword:
    filtered = df[df["tweet"].str.contains(keyword, case=False)]

    if len(filtered) > 0:
        filtered["Prediction"] = pipeline.predict(filtered["tweet"])
        st.write(filtered[["tweet", "Prediction"]])
        st.bar_chart(filtered["Prediction"].value_counts())
    else:
        st.write("No tweets found.")
