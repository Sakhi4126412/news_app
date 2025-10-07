# ================================================================
# 🧠 Fake News Detection App – Professional Version
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
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ================================================================
# 🎨 Custom Styling
# ================================================================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #e6f4ea;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #34a853;
    }
    .warning-box {
        background-color: #fef7e0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #f9ab00;
    }
    .info-box {
        background-color: #e8f0fe;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1a73e8;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

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
# 🎯 Model Dictionary
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
# 🎨 Streamlit App Layout
# ================================================================
st.markdown("<h1 class='main-header'>🧠 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ================================================================
# 🧩 Layout Columns
# ================================================================
left, right = st.columns([1, 2.5])

with left:
    st.markdown("<h2 class='sub-header'>📂 Data Upload & Configuration</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='info-box'>Upload your dataset and select the appropriate columns for analysis.</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("**Upload CSV File**", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"<div class='success-box'>✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns</div>", unsafe_allow_html=True)
            
            with st.expander("🔍 Preview Dataset", expanded=True):
                st.dataframe(df.head(3), use_container_width=True)
            
            text_col = st.selectbox("**📝 Select Text Column**", df.columns)
            target_col = st.selectbox("**🎯 Select Target Column**", df.columns)
            
            # Add model selection option
            st.markdown("**⚙️ Model Settings**")
            selected_models = st.multiselect(
                "Choose models to evaluate:",
                list(models.keys()),
                default=list(models.keys())[:4]
            )
            
            # Filter models based on selection
            filtered_models = {name: models[name] for name in selected_models}
            
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                with st.spinner("Training models... This may take a few minutes."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Preprocessing
                    progress_bar.progress(10)
                    df = preprocess_pipeline(df, text_col)
                    X = df[text_col]
                    y = df[target_col]
                    
                    # TF-IDF Vectorization
                    progress_bar.progress(30)
                    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                    X_vec = vectorizer.fit_transform(X)
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_vec, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Apply SMOTE for balancing
                    progress_bar.progress(50)
                    from collections import Counter
                    class_counts = Counter(y_train)
                    min_class_size = min(class_counts.values())
                    
                    # Adjust k_neighbors automatically
                    k_neighbors = 1 if min_class_size <= 2 else min(5, min_class_size - 1)
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    
                    try:
                        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                    except:
                        st.warning("SMOTE failed. Using original training data.")
                        X_train_res, y_train_res = X_train, y_train
                    
                    # Train and evaluate models
                    results = []
                    model_count = len(filtered_models)
                    
                    for i, (name, model) in enumerate(filtered_models.items()):
                        progress = 50 + (i / model_count) * 40
                        progress_bar.progress(int(progress))
                        
                        model.fit(X_train_res, y_train_res)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        results.append({"Model": name, "Accuracy": acc})
                    
                    progress_bar.progress(95)
                    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
                    best_model_name = results_df.iloc[0]["Model"]
                    best_acc = results_df.iloc[0]["Accuracy"]
                    best_model = filtered_models[best_model_name]
                    
                    # Retrain best model on full data
                    best_model.fit(X_train_res, y_train_res)
                    y_pred_best = best_model.predict(X_test)
                    
                    progress_bar.progress(100)
                    
                    # Store in session state
                    st.session_state['results_df'] = results_df
                    st.session_state['best_model_name'] = best_model_name
                    st.session_state['best_acc'] = best_acc
                    st.session_state['best_model'] = best_model
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred_best'] = y_pred_best
                    st.session_state['vectorizer'] = vectorizer
                    
                    st.balloons()

# ================================================================
# 🎯 Model Evaluation (Right Column)
# ================================================================
with right:
    if 'results_df' in st.session_state:
        st.markdown("<h2 class='sub-header'>📊 Model Performance Analysis</h2>", unsafe_allow_html=True)
        
        # Best model metrics
        col1, col2, col3 = st.columns(3)
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
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Models Evaluated</h3>
                <p style="font-size: 1.2rem; font-weight: bold;">{len(st.session_state['results_df'])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization section
        st.markdown("<h3 style='margin-top: 2rem;'>Model Performance Comparison</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Enhanced accuracy chart
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#1f77b4' if x != st.session_state['best_model_name'] else '#ff7f0e' for x in st.session_state['results_df']['Model']]
            
            bars = ax.barh(st.session_state['results_df']['Model'], 
                          st.session_state['results_df']['Accuracy'], 
                          color=colors, alpha=0.8)
            
            # Add value labels on bars
            for i, (v, bar) in enumerate(zip(st.session_state['results_df']['Accuracy'], bars)):
                ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{v:.2%}', ha='left', va='center', fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_xlabel('Accuracy Score', fontweight='bold')
            ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
        
        with col2:
            # Confusion matrix
            st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
            cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred_best'])
            fig, ax = plt.subplots(figsize=(4, 3.5))
            
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                       cbar_kws={'shrink': 0.8}, annot_kws={"size": 12})
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
            ax.set_title(f'{st.session_state["best_model_name"]}', fontsize=12, fontweight='bold')
            
            st.pyplot(fig)
        
        # Detailed metrics
        st.markdown("<h3 style='margin-top: 1rem;'>Detailed Classification Report</h3>", unsafe_allow_html=True)
        
        # Convert classification report to dataframe for better display
        report = classification_report(st.session_state['y_test'], st.session_state['y_pred_best'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Style the dataframe
        styled_report = report_df.style.format("{:.2f}").background_gradient(cmap='Blues', subset=pd.IndexSlice[:-1, :-1])
        st.dataframe(styled_report, use_container_width=True)
        
        # ================================================================
        # 🧪 Test Your Own Text
        # ================================================================
        st.markdown("---")
        st.markdown("<h2 class='sub-header'>🧪 Text Classification</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='info-box'>Enter text below to classify it using the best performing model.</div>", unsafe_allow_html=True)
        
        user_input = st.text_area("**Enter news text to analyze:**", height=120, 
                                 placeholder="Paste news article or headline here...")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("Analyzing text..."):
                        cleaned_input = clean_text(user_input)
                        lemmatized_input = lemmatize_text(cleaned_input)
                        X_input = st.session_state['vectorizer'].transform([lemmatized_input])
                        prediction = st.session_state['best_model'].predict(X_input)[0]
                        probability = st.session_state['best_model'].predict_proba(X_input)[0]
                        
                        # Display results
                        if prediction == 1 or str(prediction).lower() == "true":
                            st.markdown(f"""
                            <div class="warning-box" style="border-left-color: #ea4335;">
                                <h3>🚨 Prediction: FAKE NEWS</h3>
                                <p>Confidence: {max(probability):.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success-box" style="border-left-color: #34a853;">
                                <h3>✅ Prediction: REAL NEWS</h3>
                                <p>Confidence: {max(probability):.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show probability distribution
                        fig, ax = plt.subplots(figsize=(8, 2))
                        classes = st.session_state['best_model'].classes_
                        bars = ax.barh(classes, probability, color=['#34a853', '#ea4335'])
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Probability')
                        ax.set_title('Classification Confidence')
                        ax.bar_label(bars, fmt='%.2f', padding=3)
                        ax.grid(axis='x', alpha=0.3)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        st.pyplot(fig)
                else:
                    st.warning("Please enter some text to analyze.")

# ================================================================
# 📊 Dataset Statistics (if no file uploaded)
# ================================================================
else:
    with right:
        st.markdown("<h2 class='sub-header'>📋 Application Overview</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>Welcome to the Fake News Detection System</h3>
        <p>This application uses machine learning to classify news articles as real or fake.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>📁 How to Use</h4>
            <ol>
                <li>Upload a CSV dataset with text content</li>
                <li>Select the text column and target column</li>
                <li>Choose which models to evaluate</li>
                <li>Train models and analyze performance</li>
                <li>Test individual text samples</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>🔧 Technical Details</h4>
            <ul>
                <li>Uses TF-IDF for text vectorization</li>
                <li>Implements multiple ML algorithms</li>
                <li>Applies SMOTE for class balancing</li>
                <li>Provides comprehensive evaluation metrics</li>
                <li>Supports custom text classification</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Placeholder for when no data is uploaded
        st.info("👈 Please upload a CSV file to begin analysis.")

# ================================================================
# 📝 Footer
# ================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Fake News Detection System | Built with Streamlit</div>", 
    unsafe_allow_html=True
)
