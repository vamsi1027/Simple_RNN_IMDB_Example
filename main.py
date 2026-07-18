import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN
import streamlit as st


class CompatibleSimpleRNN(SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)


def load_sentiment_model():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    model_path = os.path.join(os.path.dirname(__file__), "simple_rnn_imdb.h5")

    model = load_model(
        model_path,
        custom_objects={"SimpleRNN": CompatibleSimpleRNN},
    )
    return model, word_index, reverse_word_index


MODEL, WORD_INDEX, REVERSE_WORD_INDEX = load_sentiment_model()


def decode_review(encoded_review):
    return " ".join([REVERSE_WORD_INDEX.get(i - 3, "?") for i in encoded_review])


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [WORD_INDEX.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def run_app():
    st.title("IMDB Movie Review Sentiment Analysis")
    st.write("Enter a movie review to classify it as positive or negative.")

    user_input = st.text_area("Movie Review")

    if st.button("Classify"):
        if not user_input.strip():
            st.warning("Please enter a review before classifying.")
            return

        preprocessed_input = preprocess_text(user_input)
        prediction = MODEL.predict(preprocessed_input, verbose=0)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {prediction[0][0]:.4f}")
    else:
        st.write("Please enter a movie review.")


if __name__ == "__main__":
    run_app()

