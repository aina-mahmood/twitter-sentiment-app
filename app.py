import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="✈️",
    layout="wide"
)

st.title("Twitter Sentiment Analysis ✈️")
st.caption("NLP-powered sentiment classification on airline tweets (dataset mode — stable for demos).")

@st.cache_data
def load_data():
    df = pd.read_csv("twitter airline training data.csv")
    # expected columns: text, airline_sentiment
    df = df[["text", "airline_sentiment"]].dropna()
    df.columns = ["tweet", "sentiment"]
    df["tweet"] = df["tweet"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str)
    return df

df = load_data()

@st.cache_resource
def train_model(dataframe: pd.DataFrame):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)),
        ("clf", LogisticRegression(max_iter=500))
    ])
    model.fit(dataframe["tweet"], dataframe["sentiment"])
    return model

model = train_model(df)

# --- Sidebar controls ---
st.sidebar.header("Controls")
keyword = st.sidebar.text_input("Keyword (e.g., delay, service, luggage)", "")
max_rows = st.sidebar.slider("Rows to display", 10, 200, 50, step=10)
show_examples = st.sidebar.checkbox("Show example tweets", True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset size: **{len(df):,}** tweets")

# --- Main ---
if not keyword:
    st.info("Type a keyword in the sidebar to filter tweets and see predicted sentiment.")
    st.stop()

filtered = df[df["tweet"].str.contains(keyword, case=False, na=False)].copy()

if filtered.empty:
    st.warning("No matching tweets found. Try another keyword.")
    st.stop()

filtered["prediction"] = model.predict(filtered["tweet"])

# --- Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Matches", f"{len(filtered):,}")
col2.metric("Negative", f"{(filtered['prediction']=='negative').sum():,}")
col3.metric("Neutral", f"{(filtered['prediction']=='neutral').sum():,}")
col4.metric("Positive", f"{(filtered['prediction']=='positive').sum():,}")

st.markdown("### Sentiment distribution")
counts = filtered["prediction"].value_counts()
st.bar_chart(counts)

st.markdown("### Predictions (sample)")
st.dataframe(
    filtered[["tweet", "prediction"]].head(max_rows),
    use_container_width=True
)

if show_examples:
    st.markdown("### Quick examples")
    left, right = st.columns(2)

    with left:
        st.subheader("Most likely negative 😕")
        neg = filtered[filtered["prediction"] == "negative"]["tweet"].head(5).tolist()
        if neg:
            for t in neg:
                st.write("• ", t)
        else:
            st.write("No negative examples for this keyword.")

    with right:
        st.subheader("Most likely positive 😊")
        pos = filtered[filtered["prediction"] == "positive"]["tweet"].head(5).tolist()
        if pos:
            for t in pos:
                st.write("• ", t)
        else:
            st.write("No positive examples for this keyword.")

st.markdown("---")
st.caption("Tip: Try keywords like **delay**, **cancelled**, **service**, **luggage**, **refund**.")
