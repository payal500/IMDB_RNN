import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st 

#load model
model=load_model('imdb.h5', compile=False)

word_index=imdb.get_word_index()
reverse_word_index={value: key for key, value in word_index.items() }

#step 2 : helper function to decode reviews 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#function to preprocess user input 
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#prediction function 
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment ='positive' if prediction[0][0] >0.5 else 'negative'
    return sentiment,prediction[0][0]

#streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie review to classify it as postivie or negative.')

#user input 
user_input=st.text_area('movie review')
if st.button('classify'):
    preprocess_input=preprocess_text(user_input)

    #make prediction 

    prediction=model.predict(preprocess_input)
    sentiment ='positive' if prediction[0][0] >0.5 else 'negative'
    st.write(f'sentiment: {sentiment}')
    st.write(f'prediction score : {prediction[0][0]}')
else:
    st.write("enter movie review")
