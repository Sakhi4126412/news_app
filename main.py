# app.py
# ============================================
# 📌 Fake News Detection App — Advanced (SMOTE, LIME, Confidence, 6 models)
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

import plotly.express as px
from wordcloud import WordCloud

# LIME
from lime.lime_text import LimeTextExplainer

# caching helpers
@st.cache_resource
def safe_load_spacy():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        import spacy
        return spacy.load("en_core_web_sm")

# -------------------------
# App config & UI
# -------------------------
st.set_page_config(page_title="Fake News Detection (Explainable)", layout="wide")
st.title("📰 Fake News Detection — Explainable AI Dashboard")
st.markdown("Upload a CSV with a text column and a target label (e.g. `label` with `fake` / `real`).")

# Sidebar: preprocessing options + model choices
st.sidebar.header("Preprocessing & Settings")
remove_urls = st.sidebar.checkbox("Remove URLs", value=True)
remove_punct = st.sidebar.checkbox("Remove punctuation", value=True)
remove_numbers = st.sidebar.checkbox("Remove numbers", value=False)
lowercase = st.sidebar.checkbox("Lowercase", value=True)
remove_stopwords = st.sidebar.checkbox("Remove stopwords (NLTK)", value=True)
min_df = st.sidebar.number_input("TF-IDF min_df", min_value=1, max_value=10, value=1, step=1)
max_features = st.sidebar.number_input("TF-IDF max_features", min_value=100, max_value=20000, value=5000, step=100)
use_smote = st.sidebar.checkbox("Apply SMOTE to training data", value=True)
num_top_lime_features = st.sidebar.slider("LIME features to show", 3, 20, 8)

st.sidebar.markdown("---")
st.sidebar.write("Models: Logistic, RandomForest, NaiveBayes, SVM, DecisionTree, KNN (all trained automatically)")

# -------------------------
# File upload
# -------------------------
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV to start. App expects one text column and one target column.")
    st.stop()

with st.spinner("Loading dataset..."):
    df = pd.read_csv(uploaded_file)

# quick sanity
st.subheader("Dataset preview")
st.dataframe(df.head())

# choose columns
cols = df.columns.tolist()
feature_col = st.selectbox("Select text column", cols)
target_col = st.selectbox("Select target/label column", cols)

# minimal cleaning
df = df[[feature_col, target_col]].dropna().reset_index(drop=True)
df[feature_col] = df[feature_col].astype(str)

# prevent duplicate column name issues
if df.columns.duplicated().any():
    df.columns = [f"{c}_{i}" if df.columns.duplicated()[j] else c for j, c in enumerate(df.columns)]

# -------------------------
# Text preprocessing helpers
# -------------------------
@st.cache_data
def load_stopwords():
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords")
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))

stopwords_set = load_stopwords() if remove_stopwords else set()

import emoji
def remove_emojis(text: str) -> str:
    return emoji.get_emoji_regexp().sub(r"", text)

def preprocess_text(text: str) -> str:
    s = str(text)
    if lowercase:
        s = s.lower()
    if remove_urls:
        s = re.sub(r"http\S+|www\.\S+", "", s)
    if remove_punct:
        s = s.translate(str.maketrans("", "", string.punctuation))
    if remove_numbers:
        s = re.sub(r"\d+", "", s)
    s = remove_emojis(s)
    s = re.sub(r"\s+", " ", s).strip()
    if remove_stopwords and stopwords_set:
        tokens = [t for t in s.split() if t not in stopwords_set]
        s = " ".join(tokens)
    return s

# Add a preview of preprocessing
if st.checkbox("Show preprocessing preview"):
    st.write(df[feature_col].head().apply(preprocess_text))

# -------------------------
# Phase-wise Analysis (cached where heavy)
# -------------------------
st.markdown("---")
st.subheader("Phase-wise NLP Analysis")

# Lexical
with st.expander("Lexical & Morphological"):
    df["word_count"] = df[feature_col].apply(lambda x: len(preprocess_text(x).split()))
    df["char_count"] = df[feature_col].apply(lambda x: len(preprocess_text(x)))
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="word_count", nbins=30, title="Word count distribution"), use_container_width=True)
    c2.plotly_chart(px.histogram(df, x="char_count", nbins=30, title="Character count distribution"), use_container_width=True)

# Syntactic (lazy spaCy)
with st.expander("Syntactic (POS tags)"):
    nlp = safe_load_spacy()
    @st.cache_data
    def compute_pos_counts(texts):
        tags = []
        for t in texts:
            doc = nlp(preprocess_text(t))
            tags.extend([tok.pos_ for tok in doc])
        return pd.Series(tags).value_counts().reset_index().rename(columns={"index":"POS",0:"Count"})
    pos_df = compute_pos_counts(df[feature_col].tolist())
    st.plotly_chart(px.bar(pos_df, x="POS", y="Count", title="POS distribution"), use_container_width=True)

# Semantic (wordcloud)
with st.expander("Semantic (WordCloud)"):
    @st.cache_data
    def make_wordcloud(texts):
        joined = " ".join(preprocess_text(t) for t in texts)
        wc = WordCloud(width=900, height=400, background_color="white").generate(joined)
        return wc
    wc_img = make_wordcloud(df[feature_col].tolist())
    st.image(wc_img.to_array())

# Pragmatic
with st.expander("Pragmatic (Sentence/Length)"):
    df["sent_len"] = df[feature_col].apply(lambda x: len(preprocess_text(x).split()))
    st.plotly_chart(px.histogram(df, x="sent_len", nbins=30, title="Sentence length distribution"), use_container_width=True)

# Discourse (class-wise wordcloud)
with st.expander("Discourse (Class-wise WordClouds)"):
    classes = df[target_col].unique().tolist()
    cols = st.columns(max(1, min(3, len(classes))))
    for i, cls in enumerate(classes):
        subset = df[df[target_col] == cls][feature_col].tolist()
        if subset:
            wc = make_wordcloud(subset)
            cols[i % len(cols)].markdown(f"**Class: {cls}**")
            cols[i % len(cols)].image(wc.to_array(), use_column_width=True)
        else:
            cols[i % len(cols)].warning("No samples")

# Target distribution
with st.expander("Target distribution"):
    tc = df[target_col].value_counts().reset_index()
    tc.columns = [target_col, "Count"]
    fig = px.pie(tc, names=target_col, values="Count", hole=0.4, title="Target distribution")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Modeling pipeline
# -------------------------
st.markdown("---")
st.subheader("Model training & recommendation")

X_raw = df[feature_col].astype(str).apply(preprocess_text)
y = df[target_col]

@st.cache_resource
def build_vectorizer(texts, min_df, max_features):
    tfidf = TfidfVectorizer(min_df=min_df, max_features=max_features)
    X = tfidf.fit_transform(texts)
    return tfidf, X

tfidf, X_vec = build_vectorizer(X_raw, min_df=min_df, max_features=max_features)

# safe split
class_counts = y.value_counts()
if class_counts.min() < 2:
    st.warning("Some classes have <2 samples; splitting without stratify")
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
if use_smote and len(np.unique(y_train)) > 1:
    try:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        st.info("SMOTE applied to training set")
    except Exception as e:
        st.warning(f"SMOTE failed: {e}")

# define models (ensure SVM has probability=True)
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

@st.spinner("Training models (this may take a moment)...")
@st.cache_resource
def train_all(models_dict, X_tr, y_tr, X_te, y_te):
    scores = {}
    trained = {}
    for name, m in models_dict.items():
        try:
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            acc = accuracy_score(y_te, preds)
            scores[name] = acc
            trained[name] = m
        except Exception as e:
            scores[name] = None
            trained[name] = None
    return scores, trained

scores, trained_models = train_all(models, X_train, y_train, X_test, y_test)

# show scores table + bar chart
scores_df = pd.DataFrame([(k, v if v is not None else 0.0) for k, v in scores.items()], columns=["Model", "Accuracy"])
st.dataframe(scores_df.style.format({"Accuracy": "{:.4f}"}))

fig = px.bar(scores_df, x="Model", y="Accuracy", text="Accuracy", color="Accuracy", color_continuous_scale="Viridis",
             title="Model comparison (accuracy)")
st.plotly_chart(fig, use_container_width=True)

# choose best available model
valid_scores = {k: v for k, v in scores.items() if v is not None}
if not valid_scores:
    st.error("No model trained successfully.")
    st.stop()

best_model_name = max(valid_scores, key=valid_scores.get)
best_model = trained_models[best_model_name]
st.success(f"Recommended model: **{best_model_name}** (Accuracy = {valid_scores[best_model_name]*100:.2f}%)")

# classification report for best model
with st.expander("Classification report (best model)"):
    y_pred_best = best_model.predict(X_test)
    cr = classification_report(y_test, y_pred_best, output_dict=True)
    cr_df = pd.DataFrame(cr).T
    st.dataframe(cr_df.style.format(precision=3))

# -------------------------
# User input, confidence, LIME explanation
# -------------------------
st.markdown("---")
st.subheader("Predict a single news text (with confidence & explanation)")

user_text = st.text_area("Enter news text to classify", height=150)
if st.button("Classify text"):
    if not user_text or not user_text.strip():
        st.warning("Please type some news text.")
    else:
        processed = preprocess_text(user_text)
        vec = tfidf.transform([processed])

        # confidence
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(vec)[0]
            class_labels = best_model.classes_
            conf_idx = np.argmax(probs)
            pred_label = class_labels[conf_idx]
            conf_score = probs[conf_idx]
        else:
            # fallback to decision_function -> approximate via margins
            pred_label = best_model.predict(vec)[0]
            conf_score = None

        st.markdown(f"**Prediction:** `{pred_label}`")
        if conf_score is not None:
            st.markdown(f"**Confidence:** {conf_score*100:.2f}%")
        else:
            st.markdown("**Confidence:** Not available for this model")

        # LIME explanation (wrap predict_proba pipeline)
        class_names = list(np.unique(y.astype(str)))

        def predict_proba_for_lime(texts):
            # texts: list of raw texts (not preprocessed) — LIME expects original text
            proc = [preprocess_text(t) for t in texts]
            Xt = tfidf.transform(proc)
            # if model supports predict_proba:
            if hasattr(best_model, "predict_proba"):
                return best_model.predict_proba(Xt)
            else:
                # create pseudo-probabilities using predict (1.0 for predicted class)
                preds = best_model.predict(Xt)
                probs = np.zeros((len(preds), len(class_names)))
                for i, p in enumerate(preds):
                    probs[i, class_names.index(str(p))] = 1.0
                return probs

        explainer = LimeTextExplainer(class_names=class_names)
        with st.spinner("Generating LIME explanation..."):
            try:
                exp = explainer.explain_instance(user_text, predict_proba_for_lime, num_features=num_top_lime_features)
                # show list
                lime_list = exp.as_list(label=exp.available_labels()[0] if exp.available_labels() else None)
                st.markdown("**Top contributing tokens (LIME):**")
                lime_df = pd.DataFrame(lime_list, columns=["Token", "Contribution"])
                st.dataframe(lime_df)
                # matplotlib figure
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"LIME explanation failed: {e}")

# -------------------------
# Download trained best model (optional)
# -------------------------
st.markdown("---")
st.write("You can download the TF-IDF vectorizer and the best model (joblib).")
import joblib, io
if st.button("Prepare download package (vectorizer + best model)"):
    buffer = io.BytesIO()
    package = {"tfidf": tfidf, "model": best_model}
    joblib.dump(package, "model_package.pkl")
    with open("model_package.pkl", "rb") as f:
        st.download_button("Download model_package.pkl", data=f, file_name="model_package.pkl")
