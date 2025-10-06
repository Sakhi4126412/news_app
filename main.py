# ============================================================
# 📰 Fake News Detection Dashboard (Upgraded Streamlit App)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from streamlit_extras.metric_cards import style_metric_cards

# Load SpaCy Model
nlp = spacy.load("en_core_web_sm")

# ============================================================
# 🧹 TEXT PREPROCESSING
# ============================================================
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

# ============================================================
# 🎨 STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fake News Detection Dashboard 📰",
    layout="wide",
    page_icon="🧠"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size:38px;
        font-weight:700;
        text-align:center;
        color:#1e3a8a;
    }
    .sub-title {
        text-align:center;
        color:#334155;
        font-size:18px;
        margin-bottom:30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🧠 Fake News Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze and detect fake news using NLP and Machine Learning 🚀</div>', unsafe_allow_html=True)

# ============================================================
# 📤 DATA UPLOAD SECTION
# ============================================================
uploaded_file = st.file_uploader("📁 Upload your dataset (CSV file)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    text_col = st.selectbox("📝 Select Text Column", df.columns)
    label_col = st.selectbox("🏷️ Select Label Column", df.columns)

    with st.spinner("🔄 Preprocessing text and preparing data..."):
        df["clean_text"] = df[text_col].astype(str).apply(preprocess_pipeline)

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df["clean_text"])
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # ============================================================
    # ⚙️ MODEL TRAINING
    # ============================================================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="linear"),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    }

    st.subheader("🚀 Model Training Progress")
    progress_bar = st.progress(0)
    model_results = {}

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train_res, y_train_res)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        model_results[name] = acc
        progress_bar.progress((i + 1) / len(models))

    results_df = pd.DataFrame(model_results.items(), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)

    # ============================================================
    # 🧾 METRICS CARDS
    # ============================================================
    st.markdown("### 🧾 Performance Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🪄 Best Model", results_df.iloc[0]["Model"])
    col2.metric("🎯 Best Accuracy", f"{results_df.iloc[0]['Accuracy']*100:.2f}%")
    col3.metric("📊 Models Tested", len(models))
    style_metric_cards(background_color="#f0f9ff", border_color="#3b82f6")

    # ============================================================
    # 📈 VISUALIZATIONS
    # ============================================================
    st.subheader("📊 Model Accuracy Comparison")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Accuracy", y="Model", data=results_df, palette="cool")
        plt.title("Accuracy by Model")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        wedges, texts, autotexts = ax2.pie(
            results_df["Accuracy"],
            labels=results_df["Model"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.3)
        )
        plt.title("Model Accuracy Share")
        st.pyplot(fig2)

    # ============================================================
    # 🧪 TESTING INTERFACE
    # ============================================================
    st.markdown("### 🧪 Test a News Sample")
    test_text = st.text_area("📰 Enter your news headline or article:")

    if st.button("🔍 Predict News Type"):
        best_model_name = results_df.iloc[0]["Model"]
        best_model = models[best_model_name]

        cleaned_input = preprocess_pipeline(test_text)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = best_model.predict(input_vector)[0]

        st.success(f"✅ Predicted as: **{prediction}**")
        st.info(f"Model Used: {best_model_name}")

    # ============================================================
    # 📉 CONFUSION MATRIX
    # ============================================================
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    preds_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds_best)

    st.markdown("### 📉 Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig3)

else:
    st.info("👆 Please upload a dataset to get started.")
