# ============================================================
# 🧠 Fake News Detection Dashboard - Premium Visuals Version
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.let_it_rain import rain
from streamlit_extras.switch_page_button import switch_page

# ============================================================
# 🎨 Streamlit Page Config
# ============================================================
st.set_page_config(
    page_title="Fake News Detection Dashboard 📰",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# 🌈 Custom CSS Styling (Dark + Light Themes)
# ============================================================
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        color: #1e293b;
    }
    .main-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: #0f172a;
        background: linear-gradient(90deg, #3b82f6, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 4s infinite alternate;
    }
    @keyframes pulse {
        0% {letter-spacing: 1px;}
        100% {letter-spacing: 3px;}
    }
    .sub-title {
        text-align: center;
        color: #475569;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #a5b4fc;
        background-color: #f8fafc;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #3b82f6);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 8px 16px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 🌟 Title and Subtitle
# ============================================================
st.markdown('<div class="main-title">🧠 Fake News Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Detect fake news with AI-powered NLP Models 🚀</div>', unsafe_allow_html=True)

# Add visual effect
rain(
    emoji="📰",
    font_size=20,
    falling_speed=5,
    animation_length="infinite"
)

# ============================================================
# 🧹 Text Preprocessing
# ============================================================
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def preprocess_pipeline(text):
    return lemmatize_text(clean_text(text))

# ============================================================
# 📤 Upload Dataset
# ============================================================
uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    text_col = st.selectbox("📝 Select Text Column", df.columns)
    label_col = st.selectbox("🏷️ Select Label Column", df.columns)

    with st.spinner("⚙️ Preprocessing data..."):
        df["clean_text"] = df[text_col].astype(str).apply(preprocess_pipeline)
        vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df["clean_text"])
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # ============================================================
    # ⚙️ Model Training (Improved Parameters)
    # ============================================================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=400, C=2.0),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "SVM": SVC(kernel="linear", C=2),
        "XGBoost": xgb.XGBClassifier(eval_metric="mlogloss", n_estimators=250, learning_rate=0.1)
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
    # 🧾 Summary Cards
    # ============================================================
    st.markdown("### 🧾 Performance Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Model", results_df.iloc[0]["Model"])
    col2.metric("🎯 Best Accuracy", f"{results_df.iloc[0]['Accuracy']*100:.2f}%")
    col3.metric("🧩 Models Tested", len(models))
    style_metric_cards(background_color="#eef2ff", border_color="#6366f1")

    # ============================================================
    # 📊 Visuals
    # ============================================================
    st.markdown("### 📈 Model Accuracy Comparison")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Accuracy", y="Model", data=results_df, palette="mako")
        ax.set_title("Model Accuracy (%)")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        wedges, texts, autotexts = ax2.pie(
            results_df["Accuracy"], labels=results_df["Model"],
            autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.35)
        )
        plt.title("Model Accuracy Share")
        st.pyplot(fig2)

    # ============================================================
    # 🧪 Testing Section
    # ============================================================
    st.markdown("### 🧪 Test Your Own News")
    user_input = st.text_area("✍️ Enter News Headline or Article")

    if st.button("🔍 Predict News Type"):
        best_model_name = results_df.iloc[0]["Model"]
        best_model = models[best_model_name]

        processed_input = preprocess_pipeline(user_input)
        input_vec = vectorizer.transform([processed_input])
        pred = best_model.predict(input_vec)[0]

        st.markdown(f"### ✅ Prediction: **{pred.upper()}**")
        st.caption(f"Model used: {best_model_name}")

    # ============================================================
    # 📉 Confusion Matrix
    # ============================================================
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    st.markdown("### 📉 Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig3)

else:
    st.info("👆 Upload a dataset to begin your AI-powered fake news detection journey.")
