import streamlit as st
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model
model = pickle.load(open('spam_model.pkl', 'rb'))

# Cleaning setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

# UI
st.title("📩 Spam Detection App")

msg = st.text_area("Enter your message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(msg)
        result = model.predict([cleaned])
        
        if result[0] == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")