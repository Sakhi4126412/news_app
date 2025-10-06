# ============================================
# 📌 Streamlit NLP Phase-wise Pro Analysis (Modern UI)
# ============================================

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
import seaborn as sns
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
# Evaluate models
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
st.set_page_config(page_title="Rumor Buster Pro", layout="wide", page_icon="🧠")
st.markdown(
    "<h1 style='text-align:center;color:#1f77b4;'>🧠 Rumor Buster Pro</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align:center;color:#555;'>Explore NLP features and predict labels interactively</h4>",
    unsafe_allow_html=True
)

# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head(), use_container_width=True)

    # ----------------------------
    # Configuration
    # ----------------------------
    text_col = st.selectbox("Select Text Column:", df.columns)
    target_col = st.selectbox("Select Target Column:", df.columns)
    phase = st.selectbox("Select NLP Phase:", ["Lexical & Morphological","Syntactic","Semantic","Discourse","Pragmatic"])
    run_analysis = st.button("🚀 Run Analysis")

    if run_analysis:
        with st.spinner("Processing... This may take a few seconds."):
            X = df[text_col].astype(str)
            y = df[target_col]

            # Feature extraction
            if phase=="Lexical & Morphological":
                X_processed = X.apply(lexical_preprocess)
                vectorizer = CountVectorizer()
                X_features = vectorizer.fit_transform(X_processed)
            elif phase=="Syntactic":
                X_processed = X.apply(syntactic_features)
                vectorizer = CountVectorizer()
                X_features = vectorizer.fit_transform(X_processed)
            elif phase=="Semantic":
                X_features = pd.DataFrame(X.apply(semantic_features).tolist(), columns=["polarity","subjectivity"])
            elif phase=="Discourse":
                X_processed = X.apply(discourse_features)
                vectorizer = CountVectorizer()
                X_features = vectorizer.fit_transform(X_processed)
            elif phase=="Pragmatic":
                X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(), columns=pragmatic_words)

            results = evaluate_models(X_features, y)

            # Convert results to DataFrame
            results_df = pd.DataFrame({
                "Model":[m for m in results.keys()],
                "Accuracy":[results[m]["accuracy"] for m in results.keys()]
            })
            results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

        # ----------------------------
        # Visualization with Seaborn
        # ----------------------------
        st.subheader("📊 Model Performance Overview")
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
        sns.barplot(x="Model", y="Accuracy", data=results_df, palette="Set2", ax=ax1)
        ax1.set_ylim(0, min(100, max(results_df["Accuracy"])+15))
        for idx, row in results_df.iterrows():
            ax1.text(idx, row["Accuracy"]+1, f"{row['Accuracy']}%", ha='center', fontweight='bold')
        ax1.set_title(f"Accuracy per Model - {phase}", fontsize=14, fontweight='bold')

        # Donut chart
        wedges, texts, autotexts = ax2.pie(
            results_df["Accuracy"],
            labels=results_df["Model"],
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2"),
            explode=[0.1 if i==results_df["Accuracy"].idxmax() else 0 for i in range(len(results_df))]
        )
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        ax2.add_artist(centre_circle)
        ax2.set_title("Performance Distribution", fontsize=14, fontweight='bold')
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('black')
        plt.tight_layout()
        st.pyplot(fig)

        # ----------------------------
        # Top Metrics
        # ----------------------------
        st.subheader("🏆 Top Model Benchmarks")
        best_idx = results_df["Accuracy"].idxmax()
        cols = st.columns(len(results_df))
        for idx, (model, acc) in enumerate(zip(results_df["Model"], results_df["Accuracy"])):
            with cols[idx]:
                if idx==best_idx:
                    st.metric(label=f"🥇 {model}", value=f"{acc:.1f}%", delta="Best Performance", delta_color="normal")
                else:
                    st.metric(label=model, value=f"{acc:.1f}%", delta=f"-{round(results_df.loc[best_idx,'Accuracy']-acc,1)}%", delta_color="inverse")

        # ----------------------------
        # Interactive Text Testing
        # ----------------------------
        st.subheader("📝 Test Your Text")
        user_input = st.text_area("Enter text to predict:")
        if user_input:
            best_model_name = results_df.loc[0,"Model"]
            best_model = results[best_model_name]["model"]

            if phase=="Lexical & Morphological":
                user_feat = vectorizer.transform([lexical_preprocess(user_input)])
            elif phase=="Syntactic":
                user_feat = vectorizer.transform([syntactic_features(user_input)])
            elif phase=="Semantic":
                user_feat = pd.DataFrame([semantic_features(user_input)], columns=["polarity","subjectivity"])
            elif phase=="Discourse":
                user_feat = vectorizer.transform([discourse_features(user_input)])
            elif phase=="Pragmatic":
                user_feat = pd.DataFrame([pragmatic_features(user_input)], columns=pragmatic_words)

            prediction = best_model.predict(user_feat)[0]
            st.success(f"Predicted Label: {prediction}")

else:
    st.info("👆 Upload a CSV file to start the NLP analysis.")

# ============================
# Custom CSS for Modern Look
# ============================
st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(90deg,#1f77b4,#4ECDC4);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 0px;
    }
    .stTextArea>div>textarea {
        border-radius: 8px;
        border: 2px solid #1f77b4;
        padding: 10px;
    }
    .stDataFrame {
        border-radius: 10px;
        border: 2px solid #e1e4e8;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)
