# app.py

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# --- NLTK Resource Download ---
# This block ensures all necessary NLTK data is downloaded.
# It's designed to run only once and handle potential errors gracefully.
try:
    # Check for all required resources at once
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    # If any resource is missing, download it
    print("Downloading missing NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("Downloads complete.")

# --- Load Model and Vectorizer ---
# Use st.cache_resource to load these only once and cache them for performance.
@st.cache_resource
def load_model():
    with open('optimized_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

# --- Text Preprocessing Function ---
# This function must be identical to the one used during model training.
def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join the tokens back into a string
    processed_text = ' '.join(filtered_tokens)
    
    return processed_text

# --- Main Application ---
def main():
    st.title("🎬 IMDB Movie Review Sentiment Analysis")
    st.markdown("""
    Enter a movie review below to get an instant sentiment prediction.
    The model will classify the review as either **Positive** or **Negative**.
    """)
    
    # Load the model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()
    
    # User input
    user_input = st.text_area("Enter your movie review:", height=200, placeholder="This movie was absolutely fantastic! The acting was superb...")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            # Preprocess the user input
            processed_input = preprocess_text(user_input)
            
            # Vectorize the processed input
            vectorized_input = vectorizer.transform([processed_input])
            
            # Make prediction
            prediction = model.predict(vectorized_input)[0]
            prediction_proba = model.predict_proba(vectorized_input)[0]
            
            # Display the result
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = prediction_proba[prediction] * 100
            
            st.markdown("---")
            st.subheader("Analysis Result:")
            
            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment}")
            else:
                st.error(f"Sentiment: {sentiment}")
                
            st.write(f"Confidence: {confidence:.2f}%")
            
            # Display probability breakdown
            st.write("Probability Breakdown:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Negative", f"{prediction_proba[0]*100:.2f}%")
            with col2:
                st.metric("Positive", f"{prediction_proba[1]*100:.2f}%")
        else:
            st.warning("Please enter a movie review to analyze.")

# Run the main function
if __name__ == "__main__":
    main()