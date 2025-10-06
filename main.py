# ============================================
# 📘 Rumor Buster Pro - NLP Phase-wise Analysis
# ============================================

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import numpy as np

# ============================
# SpaCy Model Download (Streamlit Cloud safe)
# ============================
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

# ============================
# Feature Extraction Functions
# ============================
def lexical_preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    doc = nlp(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Model Evaluation with SMOTE
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    # Apply SMOTE only if numeric
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except:
        X_train_res, y_train_res = X_train, y_train

    for name, model in models.items():
        try:
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = {
                "accuracy": round(acc, 2),
                "model": model,
                "report": classification_report(y_test, y_pred, zero_division=0)
            }
        except Exception as e:
            results[name] = {"accuracy": 0, "model": None, "report": str(e)}

    return results

# ============================
# Streamlit Layout
# ============================
st.set_page_config(page_title="Rumor Buster Pro", layout="wide")
st.title("🧠 Rumor Buster Pro - NLP Phase-wise Analysis")
st.markdown("#### Explore linguistic analysis and predict labels with AI models")

# File Upload
st.markdown("### 📁 Upload CSV Data")
uploaded_file = st.file_uploader("Drag & Drop CSV or Select File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### 👀 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    text_col = st.selectbox("📝 Select Text Column:", df.columns)
    target_col = st.selectbox("🎯 Select Target Column:", df.columns)
    phase = st.selectbox("🔍 Select NLP Phase:", [
        "Lexical & Morphological",
        "Syntactic",
        "Semantic",
        "Discourse",
        "Pragmatic"
    ])

    if st.button("🚀 Run Analysis"):
        with st.spinner("Processing..."):
            X = df[text_col].astype(str)
            y = df[target_col]

            # Phase feature extraction
            if phase == "Lexical & Morphological":
                X_processed = X.apply(lexical_preprocess)
                X_features = CountVectorizer().fit_transform(X_processed)
            elif phase == "Syntactic":
                X_processed = X.apply(syntactic_features)
                X_features = CountVectorizer().fit_transform(X_processed)
            elif phase == "Semantic":
                X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                          columns=["polarity", "subjectivity"])
            elif phase == "Discourse":
                X_processed = X.apply(discourse_features)
                X_features = CountVectorizer().fit_transform(X_processed)
            elif phase == "Pragmatic":
                X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                          columns=pragmatic_words)

            # Evaluate models
            results = evaluate_models(X_features, y)
            results_df = pd.DataFrame({
                "Model": [m for m in results.keys()],
                "Accuracy": [results[m]["accuracy"] for m in results.keys()]
            }).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

        # ============================
        # Visualizations
        # ============================
        st.subheader("📊 Model Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
        colors = ['#3b82f6','#10b981','#f59e0b','#ef4444']

        # Bar Chart
        bars = ax1.bar(results_df["Model"], results_df["Accuracy"], color=colors, alpha=0.8)
        best_idx = results_df["Accuracy"].idxmax()
        bars[best_idx].set_color("#22c55e")
        for i, acc in enumerate(results_df["Accuracy"]):
            ax1.text(i, acc+1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax1.set_title(f"{phase} Phase - Accuracy by Model")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, min(100, max(results_df["Accuracy"])+15))
        ax1.grid(axis="y", alpha=0.3)

        # Donut chart
        wedges, texts, autotexts = ax2.pie(
            results_df["Accuracy"],
            labels=results_df["Model"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors
        )
        centre_circle = plt.Circle((0,0),0.7,fc='white')
        ax2.add_artist(centre_circle)
        ax2.set_title("Performance Distribution")
        plt.tight_layout()
        st.pyplot(fig)

        # ============================
        # Metrics
        # ============================
        st.subheader("🏆 Best Model Performance")
        best_model_name = results_df.loc[best_idx,"Model"]
        best_model = results[best_model_name]["model"]

        cols = st.columns(len(results_df))
        for i,(model,acc) in enumerate(zip(results_df["Model"], results_df["Accuracy"])):
            with cols[i]:
                if i == best_idx:
                    st.metric(label=f"🥇 {model}", value=f"{acc:.1f}%", delta="Top Performer")
                else:
                    st.metric(label=model, value=f"{acc:.1f}%")

        # Detailed table + classification report
        st.write("### 🧾 Detailed Results")
        st.dataframe(results_df, use_container_width=True)
        st.write(f"### 📊 Classification Report - {best_model_name}")
        st.text(results[best_model_name]["report"])

        # ============================
        # Test Model Interface
        # ============================
        st.write("---")
        st.subheader("🧪 Test Your Text")
        user_input = st.text_area("Enter text to predict:", "")
        if user_input:
            with st.spinner("Predicting..."):
                if phase == "Lexical & Morphological":
                    user_feat = CountVectorizer().fit(X.apply(lexical_preprocess)).transform([lexical_preprocess(user_input)])
                elif phase == "Syntactic":
                    user_feat = CountVectorizer().fit(X.apply(syntactic_features)).transform([syntactic_features(user_input)])
                elif phase == "Semantic":
                    user_feat = pd.DataFrame([semantic_features(user_input)], columns=["polarity","subjectivity"])
                elif phase == "Discourse":
                    user_feat = CountVectorizer().fit(X.apply(discourse_features)).transform([discourse_features(user_input)])
                elif phase == "Pragmatic":
                    user_feat = pd.DataFrame([pragmatic_features(user_input)], columns=pragmatic_words)

                prediction = best_model.predict(user_feat)[0]
                st.success(f"🔮 Predicted Label: {prediction}")

else:
    st.info("👆 Upload a CSV file to start analysis.")

