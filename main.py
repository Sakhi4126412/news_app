# ============================================================
# 🧠 Fake News Detection Dashboard - Theme Toggle Edition
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
# 🌈 Sidebar: Theme Switch
# ============================================================
theme_mode = st.sidebar.radio("🌗 Choose Theme:", ["Light Mode", "Dark Mode"])
dark = theme_mode == "Dark Mode"

# ============================================================
# 🎨 Dynamic CSS Styling
# ============================================================
if dark:
    bg_color = "#0f172a"
    text_color = "#f1f5f9"
    accent = "#6366f1"
    card_bg = "#1e293b"
else:
    bg_color = "#f8fafc"
    text_color = "#1e293b"
    accent = "#3b82f6"
    card_bg = "#eef2ff"

st.markdown(
    f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .main-title {{
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: {accent};
        padding: 10px;
        margin-bottom: 5px;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
    }}
    .sub-title {{
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 30px;
    }}
    .stTextArea textarea {{
        border-radius: 12px;
        border: 2px solid {accent};
        background-color: {card_bg};
        color: {text_color};
    }}
    .stButton>button {{
        background: linear-gradient(90deg, #6366f1, #3b82f6);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 10px 18px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    }}
    .stButton>button:hover {{
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        transform: scale(1.04);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 🌟 Title Section
# ============================================================
st.markdown('<div class="main-title">🧠 Fake News Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered NLP for truth verification ⚡</div>', unsafe_allow_html=True)

rain(emoji="📰", font_size=18, falling_speed=4, animation_length="infinite")

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
# 📂 Upload Dataset (Left-aligned)
# ============================================================
left_col, right_col = st.columns([1, 2])
with left_col:
    uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    with right_col:
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
    # ⚙️ Model Training (Tuned)
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
    # 🧾 Metrics Section
    # ============================================================
    st.markdown("### 🧾 Performance Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Model", results_df.iloc[0]["Model"])
    col2.metric("🎯 Accuracy", f"{results_df.iloc[0]['Accuracy']*100:.2f}%")
    col3.metric("🧩 Models Tested", len(models))
    style_metric_cards(background_color=card_bg, border_color=accent)

    # ============================================================
    # 📈 Visualization
    # ============================================================
    st.markdown("### 📈 Model Accuracy Comparison")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis" if dark else "mako")
        ax.set_title("Model Accuracy (%)")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
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
    # 📉 Confusion Matrix (Smaller, Aligned)
    # ============================================================
    st.markdown("### 📉 Confusion Matrix")
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    fig3, ax3 = plt.subplots(figsize=(4, 3))  # Reduced size
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig3)

else:
    st.info("👈 Upload a dataset to get started.")
