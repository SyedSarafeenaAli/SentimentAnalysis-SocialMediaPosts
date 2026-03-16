import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords if not already available
nltk.download("stopwords")

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model("lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Initialize preprocessing tools (only once)
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


# -----------------------------
# Text Preprocessing Function
# -----------------------------
def clean_text(text):
    """Clean and preprocess input text."""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)


# -----------------------------
# Sentiment Prediction Function
# -----------------------------
def predict_sentiment(text):
    text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([text])

    padded_sequence = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded_sequence, verbose=0)[0][0]

    sentiment = "Positive" if prediction > 0.5 else "Negative"

    return sentiment, float(prediction)


# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":

        text = request.form.get("text")

        if text:
            result, confidence = predict_sentiment(text)

            confidence = round(confidence, 2)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)