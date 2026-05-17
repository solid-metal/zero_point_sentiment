import streamlit as st
import joblib
import re
import nltk
import random

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Load Saved Files
# -----------------------------

model = joblib.load(
    "models/sentiment_model.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

# -----------------------------
# NLP Setup
# -----------------------------

stop_words = set(
    stopwords.words("english")
)

important_negations = {
    "not",
    "no",
    "nor",
    "never"
}

stop_words = stop_words - important_negations

lemmatizer = WordNetLemmatizer()

# -----------------------------
# Text Cleaning Function
# -----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    words = [

        lemmatizer.lemmatize(word)

        for word in words

        if word not in stop_words
        and len(word) > 2
    ]

    return " ".join(words)

# -----------------------------
# Rating Generator
# -----------------------------

def generate_rating(sentiment):

    if sentiment == "Positive":

        return random.randint(8,10)

    elif sentiment == "Neutral":

        return random.randint(4,7)

    else:

        return random.randint(1,3)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title(
    "Multi-Domain Sentiment Analysis"
)

st.write(
    "Analyze reviews from movies, products, education, and restaurants."
)

# Input Box
user_input = st.text_area(
    "Enter your review:"
)

# Predict Button
if st.button("Analyze Sentiment"):

    if user_input.strip() != "":

        # Clean
        cleaned = clean_text(
            user_input
        )

        # Vectorize
        vectorized = tfidf.transform(
            [cleaned]
        )

        # Predict
        prediction = model.predict(
            vectorized
        )[0]

        # Decode
        sentiment = label_encoder.inverse_transform(
            [prediction]
        )[0]

        # Rating
        rating = generate_rating(
            sentiment
        )

        # Show Results
        st.subheader(
            f"Predicted Sentiment: {sentiment}"
        )

        st.subheader(
            f"Predicted Rating: {rating}/10"
        )

    else:

        st.warning(
            "Please enter some text."
        )