# app.py

import streamlit as st
import joblib
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
import re

# Load model and vectorizer
model = joblib.load("xgboost_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
stop_words = set(stopwords.words('english'))

# Preprocess input
def clean_text(text):
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text.strip()

def preprocess(text):
    cleaned = clean_text(text)
    tokens = [w for w in word_tokenize(cleaned) if w not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🧠 Mental Health Status Classifier")
st.write("Enter a statement and get a prediction of mental health status.")

user_input = st.text_area("Enter your mental health-related text here:")

if st.button("Classify"):
    if user_input:
        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        st.success(f"🩺 **Predicted Mental Health Status**: `{prediction}`")
    else:
        st.warning("Please enter some text before classifying.")


