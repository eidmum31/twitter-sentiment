import streamlit as st
import joblib
import re
import nltk
import contractions
import unicodedata
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Define preprocessing functions
def strip_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def remove_accented_chars(text):
    return ''.join(c for c in text if not unicodedata.combining(c))

def stopwords_removal(tokens):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def pre_process_corpus(docs):
    norm_docs = []
    for doc in docs:
        doc = doc.lower()
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        doc = word_tokenize(doc)
        doc = stopwords_removal(doc)
        norm_docs.append(doc)
    
    norm_docs = [" ".join(word) for word in norm_docs]
    return norm_docs

# Load the models and vectorizer
clf_dt = joblib.load('model/decision_tree_model.pkl')
clf_rf = joblib.load('model/random_forest_model.pkl')
clf_knn = joblib.load('model/knn_model.pkl')
clf_nb = joblib.load('model/naive_bayes_model.pkl')
clf_gru = joblib.load('model/best_model_gru.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Streamlit app
st.title("Sentiment Analysis with Multiple Models")

# Dropdown menu for selecting the model
model_options = {
    'Decision Tree': clf_dt,
    'Random Forest': clf_rf,
    'KNN': clf_knn,
    'Naive Bayes': clf_nb,
    'GRU':clf_gru
}
selected_model_name = st.selectbox("Select model for prediction:", options=list(model_options.keys()))

# Input text
text_input = st.text_area("Enter text to predict sentiment:")

if st.button("Predict"):
    if text_input:
        # Preprocess the text data
        preprocessed_texts = pre_process_corpus([text_input])
        
        # Transform the preprocessed text data using the loaded vectorizer
        X_texts = vectorizer.transform(preprocessed_texts)
        
        # Get the selected model
        selected_model = model_options[selected_model_name]
        
        # Predict sentiment using the selected model
        try:
            sentiment = selected_model.predict(X_texts)[0]
            st.write(f"{selected_model_name} prediction: {sentiment}")
        except Exception as e:
            st.error(f"Error predicting sentiment: {e}")
    else:
        st.error("Please enter some text to predict sentiment.")
