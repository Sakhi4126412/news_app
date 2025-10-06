import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import spacy
import subprocess
import sys

# ============================
# Auto-download SpaCy model
# ============================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

stop_words = nlp.Defaults.stop_words

# ============================
# Feature functions
# ============================
def lexical_preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words])

def syntactic_features(text):
    doc = nlp(text)
    return " ".join([token.pos_ for token in doc])

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split())>0])}"

pragmatic_words = ["must","should","might","could","will","?","!"]
def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Evaluate models (without SMOTE)
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }

    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)*100
            results[name] = {"accuracy": round(acc,2), "model": model}
        except Exception as e:
            results[name] = {"accuracy":0, "model":None, "error":str(e)}
    return results

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Rumor Buster Pro", layout="wide")
st.title("🧠 Rumor Buster Pro - NLP Phase-wise Analysis")
st.markdown("#### Explore linguistic analysis and predict labels with AI models")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head(), use_container_width=True)

    text_col = st.selectbox("Select Text Column:", df.columns)
    target_col = st.selectbox("Select Target Column:", df.columns)
    phase = st.selectbox("Select NLP Phase:", ["Lexical & Morphological","Syntactic","Semantic","Discourse","Pragmatic"])

    if st.button("Run Analysis"):
        with st.spinner("Processing..."):
            X = df[text_col].astype(str)
            y = df[target_col]

            if phase=="Lexical & Morphological":
                X_processed = X.apply(lexical_preprocess)
                X_features = CountVectorizer().fit_transform(X_processed)
            elif phase=="Syntactic":
                X_processed = X.apply(syntactic_features)
                X_features = CountVectorizer().fit_transform(X_processed)
            elif phase=="Semantic":
                X_features = pd.DataFrame(X.apply(semantic_features).tolist(), columns=["polarity","subjectivity"])
            elif phase=="Discourse":
                X_processed = X.apply(discourse_features)
                X_features = CountVectorizer().fit_transform(X_processed)
            elif phase=="Pragmatic":
                X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(), columns=pragmatic_words)

            results = evaluate_models(X_features, y)
            results_df = pd.DataFrame({"Model":[m for m in results.keys()],
                                       "Accuracy":[results[m]["accuracy"] for m in results.keys()]})
            results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

        # Display results
        st.subheader("Model Comparison")
        st.dataframe(results_df, use_container_width=True)

        # Interactive Test
        st.subheader("Test Your Text")
        user_input = st.text_area("Enter text to predict:")
        if user_input:
            best_model_name = results_df.loc[0,"Model"]
            best_model = results[best_model_name]["model"]

            if phase=="Lexical & Morphological":
                user_feat = CountVectorizer().fit(X.apply(lexical_preprocess)).transform([lexical_preprocess(user_input)])
            elif phase=="Syntactic":
                user_feat = CountVectorizer().fit(X.apply(syntactic_features)).transform([syntactic_features(user_input)])
            elif phase=="Semantic":
                user_feat = pd.DataFrame([semantic_features(user_input)], columns=["polarity","subjectivity"])
            elif phase=="Discourse":
                user_feat = CountVectorizer().fit(X.apply(discourse_features)).transform([discourse_features(user_input)])
            elif phase=="Pragmatic":
                user_feat = pd.DataFrame([pragmatic_features(user_input)], columns=pragmatic_words)

            prediction = best_model.predict(user_feat)[0]
            st.success(f"Predicted Label: {prediction}")
else:
    st.info("Upload a CSV file to start.")
