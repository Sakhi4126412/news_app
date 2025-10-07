# ================================================================
# 🧠 Fake News Detection App - Lite Version
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ================================================================
# 🧹 Text Preprocessing (Simplified - no spaCy)
# ================================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text

def preprocess_pipeline(df, text_col):
    df[text_col] = df[text_col].apply(clean_text)
    return df

# ================================================================
# 🎯 Model Dictionary (Simplified)
# ================================================================
models = {
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# ================================================================
# 🎨 Streamlit App Layout
# ================================================================
st.set_page_config(page_title="Fake News Detection", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🧠 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ================================================================
# 🧩 Layout Columns
# ================================================================
left, right = st.columns([1, 2])

with left:
    st.subheader("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        text_col = st.selectbox("📝 Select Text Column", df.columns)
        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... Please wait."):
                # Preprocessing
                df_clean = preprocess_pipeline(df.copy(), text_col)
                X = df_clean[text_col]
                y = df[target_col]
                
                # TF-IDF Vectorization
                vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
                X_vec = vectorizer.fit_transform(X)
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_vec, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train and evaluate models
                results = []
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        results.append({"Model": name, "Accuracy": acc})
                    except Exception as e:
                        st.warning(f"Could not train {name}: {str(e)}")
                
                if results:
                    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
                    best_model_name = results_df.iloc[0]["Model"]
                    best_acc = results_df.iloc[0]["Accuracy"]
                    best_model = models[best_model_name]
                    
                    # Store in session state
                    st.session_state['results_df'] = results_df
                    st.session_state['best_model_name'] = best_model_name
                    st.session_state['best_acc'] = best_acc
                    st.session_state['best_model'] = best_model
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred'] = best_model.predict(X_test)
                    st.session_state['vectorizer'] = vectorizer
                    
                    st.balloons()

with right:
    if 'results_df' in st.session_state:
        st.subheader("📊 Model Performance")
        
        # Display best model
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Best Model</h3>
                <p style="font-size: 1.2rem; font-weight: bold;">{st.session_state['best_model_name']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Accuracy</h3>
                <p style="font-size: 1.2rem; font-weight: bold;">{st.session_state['best_acc']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=st.session_state['results_df'], y='Model', x='Accuracy', ax=ax)
        ax.set_title('Model Accuracy Comparison')
        st.pyplot(fig)
        
        # Test your own text
        st.subheader("🧪 Test Your Own Text")
        user_input = st.text_area("Enter news text:", height=100)
        
        if st.button("🔍 Predict") and user_input.strip():
            cleaned_input = clean_text(user_input)
            X_input = st.session_state['vectorizer'].transform([cleaned_input])
            prediction = st.session_state['best_model'].predict(X_input)[0]
            
            if prediction == 1 or str(prediction).lower() in ['fake', 'false']:
                st.error(f"🚨 Prediction: FAKE NEWS")
            else:
                st.success(f"✅ Prediction: REAL NEWS")

# Display instructions if no file uploaded
elif not uploaded_file:
    with right:
        st.info("👈 Please upload a CSV file to begin analysis.")
        st.markdown("""
        ### 📋 How to use:
        1. Upload a CSV file with text data
        2. Select the text column and target column  
        3. Click 'Train Models' to build classifiers
        4. View model performance and test new text
        """)
