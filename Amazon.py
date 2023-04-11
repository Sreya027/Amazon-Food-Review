import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


# Instantiate the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Define the function to calculate the sentiment score
def get_sentiment_score(text):
    return sia.polarity_scores(text)['compound']

# Define the Streamlit app
def app():
    st.title("Sentiment Analysis on Amazon Food Reviews")



    # Get the user input for the review text
    st.header("Enter your review text below:")
    review = st.text_input("", "")

    # Calculate the sentiment score
    if review:
        sentiment_score = get_sentiment_score(review)

        # Display the sentiment score
        st.header("Sentiment Score:")
        st.write(sentiment_score)

        # Display the sentiment label
        if sentiment_score >= 0.05:
            sentiment_label = "Positive"
        elif sentiment_score <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        st.header("Sentiment Label:")
        st.write(sentiment_label)

# Run the app
if __name__ == '__main__':
    app()
