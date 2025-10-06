# ============================================
# 📘 NLP Phase-wise Analysis App (Enhanced Pro)
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
# Load SpaCy Model
# ============================
nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

# ============================
# Phase Feature Extractors
# ============================
def lexical_preprocess(text):
    """Tokenization + Stopwords removal + Lemmatization"""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    """POS tagging features"""
    doc = nlp(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    """Sentiment polarity and subjectivity"""
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    """Sentence count and first words"""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    """Count pragmatic markers"""
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Train & Evaluate with SMOTE
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

    # Apply SMOTE only if the data is numeric
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    except:
        # For non-numeric (e.g., sparse matrix) cases
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
# Streamlit UI Layout
# ============================
st.set_page_config(page_title="NLP Phase Analyzer", layout="wide")
st.title("🧠 **Rumor Buster Pro - NLP Phase-wise Analysis**")
st.markdown("#### 💬 Explore different levels of linguistic analysis using AI models")

# File upload section
st.markdown("### 📁 Upload Your CSV File")
uploaded_file = st.file_uploader("Drag & Drop or Choose CSV", type=["csv"])

# ======================================
# MAIN APP LOGIC
# ======================================
if uploaded_file:
    st.success("✅ File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    st.write("### 👀 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    text_col = st.selectbox("📝 Select Text Column:", df.columns)
    target_col = st.selectbox("🎯 Select Target Column:", df.columns)

    phase = st.selectbox("🔍 Choose NLP Phase for Analysis:", [
        "Lexical & Morphological",
        "Syntactic",
        "Semantic",
        "Discourse",
        "Pragmatic"
    ])

    if st.button("🚀 Run Phase-wise Analysis"):
        with st.spinner("Crunching the linguistic universe..."):
            X = df[text_col].astype(str)
            y = df[target_col]

            # Phase-specific feature extraction
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
        st.write("---")
        st.subheader("📈 Model Comparison")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

        # Bar Chart
        bars = ax1.bar(results_df["Model"], results_df["Accuracy"], color=colors, alpha=0.8)
        best_idx = results_df["Accuracy"].idxmax()
        bars[best_idx].set_color("#22c55e")

        for i, acc in enumerate(results_df["Accuracy"]):
            ax1.text(i, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax1.set_title(f"{phase} Phase - Accuracy by Model", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, min(100, max(results_df["Accuracy"]) + 15))
        ax1.grid(axis="y", alpha=0.3)

        # Donut Chart
        wedges, texts, autotexts = ax2.pie(
            results_df["Accuracy"],
            labels=results_df["Model"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax2.add_artist(centre_circle)
        ax2.set_title("Performance Distribution", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        # ============================
        # Metrics Summary
        # ============================
        st.subheader("🏆 Model Performance Summary")
        best_model_name = results_df.loc[best_idx, "Model"]
        best_model = results[best_model_name]["model"]

        cols = st.columns(len(results_df))
        for i, (model, acc) in enumerate(zip(results_df["Model"], results_df["Accuracy"])):
            with cols[i]:
                if i == best_idx:
                    st.metric(label=f"🥇 {model}", value=f"{acc:.1f}%", delta="Top Performer")
                else:
                    st.metric(label=model, value=f"{acc:.1f}%")

        # Detailed results table
        st.write("### 🧾 Detailed Results")
        st.dataframe(results_df, use_container_width=True)

        # Classification report for best model
        st.write("### 📊 Classification Report (Best Model)")
        st.text(results[best_model_name]["report"])

        # ============================
        # Test Interface
        # ============================
        st.write("---")
        st.subheader("🧪 Try It Yourself")

        user_input = st.text_area("Enter text to test the trained model:", "")
        if user_input:
            with st.spinner("Analyzing your text..."):
                if phase == "Lexical & Morphological":
                    user_feat = CountVectorizer().fit(X.apply(lexical_preprocess)).transform([lexical_preprocess(user_input)])
                elif phase == "Syntactic":
                    user_feat = CountVectorizer().fit(X.apply(syntactic_features)).transform([syntactic_features(user_input)])
                elif phase == "Semantic":
                    user_feat = pd.DataFrame([semantic_features(user_input)], columns=["polarity", "subjectivity"])
                elif phase == "Discourse":
                    user_feat = CountVectorizer().fit(X.apply(discourse_features)).transform([discourse_features(user_input)])
                elif phase == "Pragmatic":
                    user_feat = pd.DataFrame([pragmatic_features(user_input)], columns=pragmatic_words)

                prediction = best_model.predict(user_feat)[0]
                st.success(f"🔮 **Predicted Label:** {prediction}")

else:
    st.info("👆 Upload a CSV file to begin your NLP phase-wise exploration.")

# ============================
# Styling
# ============================
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
    }
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)
