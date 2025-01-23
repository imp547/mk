# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:25:41 2025

@author: mk
"""
#import libaries
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer



#load data
with open("fake_news_model.pkl","rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl","rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
    
    
#stream lit
st.title("fake News Dedection")
st.write("lets predict the news intergrity")

#input
news_input =st.text_area("put your text here",height=200)

#button
if st.button("Analyse News"):
    if not news_input.strip():
        st.warning("Please enter news to analyse")
    else:
        news_vector = vectorizer.transform([news_input])
        
        prediction = model.predict(news_vector)[0]
        
        if prediction == 1:
            st.success("this news is True")
        else:
            st.error("its a fake news")
            
