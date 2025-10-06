# ============================================
# 🚀 Fake News Detection App - Enhanced Version
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# ============================================
# 🧹 Text Preprocessing
# ============================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def preprocess_pipeline(text):
    text = clean_text(text)
    text = lemmatize_text(text)
    return text

# ============================================
# 🌐 Streamlit UI
# ============================================
st.set_page_config(page_title="📰 Fake News Detection App", layout="wide")
st.title("🧠 Fake News Detection using NLP & ML")
st.markdown("### Detect Fake News with Machine Learning Models (Enhanced Accuracy)")

uploaded_file = st.file_uploader("📤 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Expect columns like ['text', 'label']
    text_col = st.selectbox("📝 Select Text Column", df.columns)
    label_col = st.selectbox("🏷️ Select Label Column", df.columns)

    with st.spinner("🧠 Preprocessing data..."):
        df['clean_text'] = df[text_col].astype(str).apply(preprocess_pipeline)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['clean_text'])
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Handle imbalance
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # ============================================
    # ⚙️ Model Selection & Training
    # ============================================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="linear"),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    }

    st.subheader("⚙️ Model Training Progress")
    progress_bar = st.progress(0)
    model_results = {}

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train_res, y_train_res)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        model_results[name] = acc
        progress_bar.progress((i + 1) / len(models))
    
    # ============================================
    # 📈 Results Visualization
    # ============================================
    st.subheader("📊 Model Performance Comparison")
    results_df = pd.DataFrame(list(model_results.items()), columns=["Model", "Accuracy"])
    st.dataframe(results_df.sort_values(by="Accuracy", ascending=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis", ax=ax)
    plt.title("Model Accuracy Comparison")
    st.pyplot(fig)

    # Donut Chart
    fig2, ax2 = plt.subplots()
    wedges, texts, autotexts = ax2.pie(
        results_df["Accuracy"],
        labels=results_df["Model"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.3)
    )
    st.pyplot(fig2)

    # ============================================
    # 🧪 Testing Interface
    # ============================================
    st.subheader("🧪 Try It Yourself")
    test_text = st.text_area("Enter News Text to Test:")

    if st.button("🔍 Predict"):
        best_model_name = results_df.iloc[results_df["Accuracy"].idxmax()]["Model"]
        best_model = models[best_model_name]

        cleaned_input = preprocess_pipeline(test_text)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = best_model.predict(input_vector)[0]

        st.success(f"✅ Predicted as: **{prediction}**")
        st.info(f"Model Used: {best_model_name}")

    # ============================================
    # 📊 Confusion Matrix for Best Model
    # ============================================
    best_model_name = results_df.iloc[results_df["Accuracy"].idxmax()]["Model"]
    best_model = models[best_model_name]
    preds_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds_best)

    st.subheader("📉 Confusion Matrix for Best Model")
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    st.pyplot(fig3)

    st.markdown(f"### 🏆 Best Model: {best_model_name} with Accuracy: **{results_df['Accuracy'].max():.2f}**")

else:
    st.info("👆 Upload your dataset above to start analysis.")
