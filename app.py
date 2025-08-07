

# import librarues 
import streamlit as st
import joblib
import tensorflow as tf
import re
import string
import pandas as pd
import numpy as np

#load model and vectorizer
logreg = joblib.load('LR_model.pkl')
dl = tf.keras.models.load_model('best_model1.keras')
vect = joblib.load('TfIdf_vect.pkl')

# clean function
def clean(text):

  # lower case
  text = text.lower()
    
  # save the pattern for the punctuation and special characters in regex
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    
  # replace the punctuation with nothing 
  text = re_punc.sub(' ', text)

  # remove extra special characters not handled above
  text = re.sub(r'[’,-, @]',' ', text) 

  # remove links
  text = re.sub(r'http\S+|www.\S+', ' ',text)

  # remove html tags
  text = re.sub(r"<.*?>", ' ',text)

  return text
   
def remove_stopwords(text):
    if 'IGBO_STOPWORDS' not in st.session_state:
        file = open("IGBO_STOPWORDS.txt", 'r', encoding='utf-8')
        st.session_state.IGBO_STOPWORDS = [line.strip() for line in file.readlines()]
        file.close()  # tokenize by whitespace
    filtered_words = [word for word in text if word not in st.session_state.IGBO_STOPWORDS]
    return filtered_words
    
# app title
st.title("Kedu?")
st.subheader("A sentiment analysis app for Igbo texts")

# read user feedback input 
input_text = st.text_area("Kedu ka ụbọchị gị na-aga? Tinye uche obi gị ebe a ka m nyochaa ya", "Type how you feel here...")

# predict button
predict1 = st.button("Jiri LogReg Model Nyochaa")
predict2 = st.button("Jiri Deep Model Nyochaa")

if input_text != "":
    st.session_state.prediction = None  # Reset prediction when user types
    # Transform the input text using the vectorizer
    st.session_state.li = remove_stopwords(clean(input_text).split())
if predict1:
    # Make prediction
    st.session_state.prediction = logreg.predict(vect.transform(st.session_state.li).toarray())[0]
if predict2:
    pred = dl.predict(vect.transform(st.session_state.li).toarray(),verbose= 0)[0]
    st.session_state.prediction = np.argmax(pred)
                                                    

# Display results based on prediction
if st.session_state.prediction is not None:
    if st.session_state.prediction == 0:
        st.markdown(f"Negative")
        st.markdown("![unhappy](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZWI1Nnl5ZHA5cXYxeDcwdHYwOHc1MzJlaGd6ZDB3M2w4NjFmbW4ydiZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/zIt1sZXSYsPpCIBPkC/giphy.gif)")
    elif st.session_state.prediction == 1:
        st.markdown(f"Neutral")
        st.markdown("![neutral](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjMybzRoOWc0aGZjODRodHNjZWlyNXo0a3V6YnB1c2U5MG15YmdmcSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/vTQggir5EQzuVoBiJt/giphy.gif)")
    else:
        st.markdown(f"Positive")
        st.markdown("![happy](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExczk4eDcwdHV3aGh1czRzNjFvNjlxYTY0ZXpra3lla3ByNDFpcG5zMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9ZQ/QWvra259h4LCvdJnxP/giphy.gif)")
