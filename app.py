import streamlit as st
import pickle
import time

# Load the model and vectorizer
model = pickle.load(open('twitter_sentiment_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.title(' Sentiment Analysis using Twitter Data')

# Input tweet from user
tweet = st.text_input('Enter your text')

submit = st.button('Predict')

if submit:
    start = time.time()

    # Transform the input text using the TfidfVectorizer
    tweet_transformed = tfidf_vectorizer.transform([tweet])

    # Make prediction
    prediction = model.predict(tweet_transformed)

    end = time.time()
    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')

    # Display prediction result
    st.write('Predicted Sentiment:', prediction[0])
