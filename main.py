import streamlit as st
import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from textblob import TextBlob
from nltk import pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize NLP tools
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/fake_news.csv")  # Must contain 'text' and 'label'
    return df.dropna()

# Linguistic analysis
def analyze_text(text):
    doc = nlp(text)
    blob = TextBlob(text)
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    return {
        "Lexical Diversity": round(len(set(tokens)) / len(tokens), 3),
        "Sentiment (TextBlob)": round(blob.sentiment.polarity, 3),
        "Sentiment (VADER)": round(sia.polarity_scores(text)['compound'], 3),
        "Named Entities": [(ent.text, ent.label_) for ent in doc.ents],
        "POS Tags": pos_tags[:10],
        "Syntactic Dependencies": [(token.text, token.dep_) for token in doc[:10]],
        "Pragmatic Intent": "Opinion" if blob.sentiment.polarity != 0 else "Neutral",
        "Discourse Markers": [token.text for token in doc if token.dep_ == "mark"]
    }

# Train model
def train_model(model_name, X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)

    models = {
        "SVM": SVC(),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier()
    }

    model = models[model_name]
    model.fit(X_res, y_res)
    return model

# App UI
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection App")
st.markdown("Detect fake news using machine learning and linguistic analysis.")

df = load_data()

# Sidebar
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose ML Model", ["SVM", "Naive Bayes", "Logistic Regression", "Decision Tree"])

# Text input
text_input = st.text_area("Enter news text for analysis and prediction:")

if text_input:
    st.subheader("🔍 Linguistic Analysis")
    analysis = analyze_text(text_input)
    for key, value in analysis.items():
        st.write(f"**{key}:** {value}")

    st.subheader("🧪 Prediction")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    model = train_model(model_choice, X, y)
    input_vec = vectorizer.transform([text_input])
    prediction = model.predict(input_vec)[0]

    st.success(f"Prediction: {'Fake News' if prediction == 1 else 'Real News'}")

    st.subheader("📈 Model Performance")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model_perf = train_model(model_choice, X_train, y_train)
    y_pred = model_perf.predict(X_test)
    st.text(classification_report(y_test, y_pred))
