# ================================================================
# 🧠 Fake News Detection App – Final Version (SMOTE Fixed)
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
import xgboost as xgb
import spacy

# ================================================================
# ⚙️ Load SpaCy Model
# ================================================================
nlp = spacy.load("en_core_web_sm")

# ================================================================
# 🧹 Text Preprocessing
# ================================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def preprocess_pipeline(df, text_col):
    df[text_col] = df[text_col].astype(str).apply(clean_text).apply(lemmatize_text)
    return df

# ================================================================
# 🎯 Models
# ================================================================
models = {
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=300),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
}

# ================================================================
# 🎨 Streamlit Setup
# ================================================================
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.markdown("<h1 style='text-align:center;'>🧠 Fake News Detection & Model Evaluation</h1>", unsafe_allow_html=True)
st.markdown("---")

# ================================================================
# 🧩 Layout
# ================================================================
left, right = st.columns([1, 2.5])

with left:
    st.subheader("📂 Upload & Configure")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(3))

        text_col = st.selectbox("📝 Select Text Column", df.columns)
        target_col = st.selectbox("🎯 Select Target Column", df.columns)

        if st.button("🚀 Train Models"):
            with st.spinner("Training models... please wait ⏳"):
                df = preprocess_pipeline(df, text_col)
                X = df[text_col]
                y = df[target_col]

                # Vectorization
                vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                X_vec = vectorizer.fit_transform(X)

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_vec, y, test_size=0.2, random_state=42
                )

                # ✅ Safe SMOTE (auto k_neighbors)
                class_counts = Counter(y_train)
                min_class_size = min(class_counts.values())
                k_neighbors = 1 if min_class_size <= 2 else min(5, min_class_size - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

                # Train & Evaluate
                results = []
                for name, model in models.items():
                    model.fit(X_train_res, y_train_res)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    results.append({"Model": name, "Accuracy": acc})

                results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
                best_model_name = results_df.iloc[0]["Model"]
                best_acc = results_df.iloc[0]["Accuracy"]
                best_model = models[best_model_name]
                best_model.fit(X_train_res, y_train_res)
                y_pred_best = best_model.predict(X_test)

            # ================================================================
            # 📊 Evaluation Section
            # ================================================================
            with right:
                st.subheader("📊 Model Evaluation")
                st.markdown(f"**🏆 Best Model:** `{best_model_name}` with **Accuracy:** `{best_acc:.2%}`")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 Model Accuracy Comparison")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.barplot(data=results_df, x="Accuracy", y="Model", palette="Blues_r", ax=ax)
                    ax.set_xlim(0, 1)
                    ax.set_title("Model Accuracy (%)", fontsize=11)
                    st.pyplot(fig)

                with col2:
                    st.subheader("📉 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred_best)
                    fig, ax = plt.subplots(figsize=(3.2, 2.8))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax, cbar=False)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title("Confusion Matrix", fontsize=11)
                    st.pyplot(fig)

                st.subheader("📋 Classification Report")
                st.text(classification_report(y_test, y_pred_best))

            # ================================================================
            # 🧪 Test Your Own Text
            # ================================================================
            st.markdown("---")
            st.subheader("🧪 Test Your Own Text")
            user_input = st.text_area("Enter a news headline or paragraph:", height=100)

            if st.button("🔍 Predict"):
                if user_input.strip():
                    cleaned_input = clean_text(user_input)
                    lemmatized_input = lemmatize_text(cleaned_input)
                    X_input = vectorizer.transform([lemmatized_input])
                    prediction = best_model.predict(X_input)[0]
                    st.success(f"🧾 **Prediction:** {prediction}")
                else:
                    st.warning("⚠️ Please enter some text to analyze.")
    else:
        st.warning("⬅️ Please upload a CSV file to begin analysis.")
