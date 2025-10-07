# ================================================================
# 🧠 Fake News Detection App - Fixed Version with Error Handling
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ================================================================
# 🎨 Page Configuration & Custom Styling
# ================================================================
st.set_page_config(
    page_title="Fake News Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-left: 5px solid #667eea;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# 🧹 Text Preprocessing Functions
# ================================================================
def clean_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_pipeline(df, text_col):
    """Apply preprocessing pipeline to dataframe"""
    df_clean = df.copy()
    df_clean['cleaned_text'] = df_clean[text_col].apply(clean_text)
    df_clean['text_length'] = df_clean['cleaned_text'].apply(len)
    df_clean['word_count'] = df_clean['cleaned_text'].apply(lambda x: len(x.split()))
    return df_clean

# ================================================================
# 📊 Visualization Functions
# ================================================================
def create_wordcloud(texts, title):
    """Create word cloud visualization"""
    try:
        # Filter out empty texts
        texts = [text for text in texts if len(text) > 0]
        if not texts:
            return None
            
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(' '.join(texts))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None

def plot_class_distribution(y):
    """Plot class distribution"""
    try:
        class_counts = pd.Series(y).value_counts()
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title="Class Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    except Exception as e:
        st.error(f"Error plotting class distribution: {str(e)}")
        return None

# ================================================================
# 🎯 Model Training & Evaluation
# ================================================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}

def train_and_evaluate_models(X_train, X_test, y_train, y_test, selected_models):
    """Train and evaluate multiple models with error handling"""
    results = []
    
    for name in selected_models:
        try:
            model = models[name]
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Model_Object': model,
                'Predictions': y_pred,
                'Probabilities': y_pred_proba
            })
            
        except Exception as e:
            st.warning(f"❌ Failed to train {name}: {str(e)}")
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# ================================================================
# 🚀 Streamlit App Layout
# ================================================================
st.markdown("<h1 class='main-header'>🔍 Advanced Fake News Detection System</h1>", unsafe_allow_html=True)

# Initialize session state with default values
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'text_col' not in st.session_state:
    st.session_state.text_col = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

# ================================================================
# 📁 Sidebar - Navigation & Configuration
# ================================================================
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    app_section = st.radio(
        "Choose Section:",
        ["📊 Data Overview", "🔧 Preprocessing", "📈 NLP Analysis", "🤖 Model Training", "🧪 Prediction"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("## ⚙️ Configuration")
    
    # Model settings
    st.markdown("### Model Settings")
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    vectorizer_type = st.selectbox("Vectorization Method", ["TF-IDF", "Count Vectorizer"])
    max_features = st.slider("Max Features", 100, 5000, 2000)
    
    st.markdown("---")
    st.markdown("### 📊 Model Selection")
    selected_models = st.multiselect(
        "Choose models to train:",
        list(models.keys()),
        default=["Random Forest", "Logistic Regression", "Naive Bayes"]
    )

# ================================================================
# 📊 SECTION 1: Data Overview & Upload
# ================================================================
if app_section == "📊 Data Overview":
    st.markdown("<h2 class='section-header'>📁 Data Upload & Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Upload Dataset")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv", help="Upload your dataset in CSV format")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.raw_data = df
                
                st.markdown("### 🎯 Column Selection")
                text_col = st.selectbox("Select Text Column", df.columns)
                target_col = st.selectbox("Select Target Column", df.columns)
                
                if st.button("🚀 Load Dataset", type="primary", use_container_width=True):
                    st.session_state.text_col = text_col
                    st.session_state.target_col = target_col
                    st.success("✅ Dataset loaded successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        if st.session_state.raw_data is not None:
            df = st.session_state.raw_data
            text_col = st.session_state.text_col
            target_col = st.session_state.target_col
            
            # Data overview cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Total Samples</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: #667eea;">{len(df):,}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏷️ Classes</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: #764ba2;">{df[target_col].nunique()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝 Text Column</h3>
                    <p style="font-size: 1.2rem; font-weight: bold; color: #2c3e50;">{text_col}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Target Column</h3>
                    <p style="font-size: 1.2rem; font-weight: bold; color: #2c3e50;">{target_col}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            tab1, tab2, tab3 = st.tabs(["First 5 Rows", "Last 5 Rows", "Statistics"])
            
            with tab1:
                st.dataframe(df.head(), use_container_width=True)
            
            with tab2:
                st.dataframe(df.tail(), use_container_width=True)
            
            with tab3:
                st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("📁 Please upload a CSV file to get started.")

# ================================================================
# 🔧 SECTION 2: Text Preprocessing
# ================================================================
elif app_section == "🔧 Preprocessing":
    st.markdown("<h2 class='section-header'>🔧 Text Preprocessing Pipeline</h2>", unsafe_allow_html=True)
    
    if st.session_state.raw_data is None:
        st.warning("⚠️ Please upload and configure your dataset in the 'Data Overview' section first.")
    else:
        df = st.session_state.raw_data
        text_col = st.session_state.text_col
        target_col = st.session_state.target_col
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Preprocessing Settings")
            
            if st.button("🔄 Apply Preprocessing", type="primary", use_container_width=True):
                with st.spinner("Processing text data..."):
                    try:
                        # Apply preprocessing
                        processed_df = preprocess_pipeline(df, text_col)
                        st.session_state.processed_data = processed_df
                        
                        # Initialize vectorizer
                        if vectorizer_type == "TF-IDF":
                            st.session_state.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
                        else:
                            st.session_state.vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
                        
                        st.success("✅ Preprocessing completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Preprocessing failed: {str(e)}")
        
        with col2:
            if st.session_state.processed_data is not None:
                processed_df = st.session_state.processed_data
                
                # Show preprocessing results
                st.markdown("### 📋 Preprocessing Results")
                
                tab1, tab2 = st.tabs(["Original vs Cleaned", "Text Statistics"])
                
                with tab1:
                    comparison_df = pd.DataFrame({
                        'Original Text': df[text_col].head(10),
                        'Cleaned Text': processed_df['cleaned_text'].head(10)
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                
                with tab2:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_length = processed_df['text_length'].mean()
                        st.metric("Average Text Length", f"{avg_length:.1f} chars")
                    with col2:
                        avg_words = processed_df['word_count'].mean()
                        st.metric("Average Word Count", f"{avg_words:.1f} words")
                    with col3:
                        empty_texts = processed_df[processed_df['cleaned_text'].str.len() == 0].shape[0]
                        st.metric("Empty Texts", empty_texts)
            else:
                st.info("🔄 Click 'Apply Preprocessing' to clean and prepare your text data.")

# ================================================================
# 📈 SECTION 3: NLP Analysis
# ================================================================
elif app_section == "📈 NLP Analysis":
    st.markdown("<h2 class='section-header'>📈 NLP Phase-wise Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please preprocess your data in the 'Preprocessing' section first.")
    else:
        processed_df = st.session_state.processed_data
        target_col = st.session_state.target_col
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 Distribution Analysis", "☁️ Word Clouds", "📈 Feature Analysis"])
        
        with tab1:
            st.markdown("### 📊 Data Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Class distribution
                fig = plot_class_distribution(processed_df[target_col])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not generate class distribution chart")
            
            with col2:
                # Text length distribution
                try:
                    fig = px.histogram(
                        processed_df, x='text_length', 
                        color=target_col if target_col in processed_df.columns else None,
                        title="Text Length Distribution",
                        nbins=50,
                        opacity=0.7
                    )
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating histogram: {str(e)}")
        
        with tab2:
            st.markdown("### ☁️ Word Cloud Analysis")
            col1, col2 = st.columns(2)
            
            # Get classes dynamically
            classes = processed_df[target_col].unique()
            
            with col1:
                if len(classes) >= 1:
                    class1_texts = processed_df[processed_df[target_col] == classes[0]]['cleaned_text']
                    fig = create_wordcloud(class1_texts, f"Word Cloud - {classes[0]}")
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info(f"No sufficient text data for {classes[0]}")
            
            with col2:
                if len(classes) >= 2:
                    class2_texts = processed_df[processed_df[target_col] == classes[1]]['cleaned_text']
                    fig = create_wordcloud(class2_texts, f"Word Cloud - {classes[1]}")
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info(f"No sufficient text data for {classes[1]}")
        
        with tab3:
            st.markdown("### 📈 Feature Analysis")
            
            try:
                # Most frequent words
                all_text = ' '.join(processed_df['cleaned_text'])
                words = all_text.split()
                word_freq = Counter(words)
                common_words = word_freq.most_common(20)
                
                words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
                fig = px.bar(words_df, x='Frequency', y='Word', orientation='h',
                            title="Top 20 Most Frequent Words")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in feature analysis: {str(e)}")

# ================================================================
# 🤖 SECTION 4: Model Training
# ================================================================
elif app_section == "🤖 Model Training":
    st.markdown("<h2 class='section-header'>🤖 Machine Learning Model Training</h2>", unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please preprocess your data first.")
    elif not selected_models:
        st.warning("⚠️ Please select at least one model to train.")
    else:
        processed_df = st.session_state.processed_data
        target_col = st.session_state.target_col
        vectorizer = st.session_state.vectorizer
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Training Configuration")
            
            if st.button("🎯 Train Models", type="primary", use_container_width=True):
                with st.spinner("Training models... This may take a few minutes."):
                    try:
                        # Prepare features and labels
                        X = vectorizer.fit_transform(processed_df['cleaned_text'])
                        y = processed_df[target_col]
                        
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42, stratify=y
                        )
                        
                        # Train models
                        results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, selected_models)
                        
                        if not results_df.empty:
                            st.session_state.trained_models = results_df
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            st.success(f"✅ Successfully trained {len(results_df)} models!")
                        else:
                            st.error("❌ No models were successfully trained. Please check your data and model selection.")
                            
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
        
        with col2:
            if (st.session_state.trained_models is not None and 
                not st.session_state.trained_models.empty):
                
                results_df = st.session_state.trained_models
                
                # Display results
                st.markdown("### 📊 Model Performance")
                
                # Best model card - WITH ERROR HANDLING
                if len(results_df) > 0:
                    best_model = results_df.iloc[0]
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🏆 Best Performing Model</h3>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{best_model['Model']}</p>
                        <p style="font-size: 1.2rem;">Accuracy: <strong>{best_model['Accuracy']:.2%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Model comparison chart
                    fig = px.bar(
                        results_df, 
                        x='Accuracy', 
                        y='Model',
                        orientation='h',
                        title="Model Accuracy Comparison",
                        color='Accuracy',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed metrics
                    st.markdown("### 📈 Detailed Evaluation")
                    best_model_obj = best_model['Model_Object']
                    y_pred = best_model['Predictions']
                    y_test = st.session_state.y_test
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confusion Matrix
                        cm = confusion_matrix(y_test, y_pred)
                        fig = px.imshow(
                            cm,
                            text_auto=True,
                            color_continuous_scale='Blues',
                            title=f"Confusion Matrix - {best_model['Model']}",
                            labels=dict(x="Predicted", y="Actual", color="Count")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Classification Report
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)
                else:
                    st.warning("No models were successfully trained.")
            else:
                st.info("🎯 Click 'Train Models' to start training your selected models.")

# ================================================================
# 🧪 SECTION 5: Prediction
# ================================================================
elif app_section == "🧪 Prediction":
    st.markdown("<h2 class='section-header'>🧪 Real-time Text Prediction</h2>", unsafe_allow_html=True)
    
    if (st.session_state.trained_models is None or 
        st.session_state.trained_models.empty):
        st.warning("⚠️ Please train models in the 'Model Training' section first.")
    else:
        results_df = st.session_state.trained_models
        
        # Check if we have a best model
        if len(results_df) == 0:
            st.error("❌ No trained models available. Please train models first.")
        else:
            best_model = results_df.iloc[0]
            vectorizer = st.session_state.vectorizer
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📝 Input Text for Analysis")
                
                user_text = st.text_area(
                    "Enter news text to analyze:",
                    height=200,
                    placeholder="Paste news article or headline here..."
                )
                
                if st.button("🔍 Analyze Text", type="primary", use_container_width=True) and user_text.strip():
                    with st.spinner("Analyzing text..."):
                        try:
                            # Preprocess input text
                            cleaned_text = clean_text(user_text)
                            
                            # Vectorize
                            X_input = vectorizer.transform([cleaned_text])
                            
                            # Make prediction
                            prediction = best_model['Model_Object'].predict(X_input)[0]
                            probability = best_model['Model_Object'].predict_proba(X_input)[0]
                            
                            # Store results
                            st.session_state.prediction = prediction
                            st.session_state.probability = probability
                            st.session_state.cleaned_text = cleaned_text
                            
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
            
            with col2:
                if 'prediction' in st.session_state:
                    prediction = st.session_state.prediction
                    probability = st.session_state.probability
                    cleaned_text = st.session_state.cleaned_text
                    
                    st.markdown("### 📊 Prediction Results")
                    
                    # Prediction card
                    confidence = max(probability)
                    predicted_class = prediction
                    
                    # Get class names from the model
                    model_classes = best_model['Model_Object'].classes_
                    
                    if predicted_class == 1 or str(predicted_class).lower() in ['fake', 'false']:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
                                    padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                            <h2 style="margin: 0; font-size: 2rem;">🚨 FAKE NEWS</h2>
                            <p style="font-size: 1.2rem; margin: 1rem 0;">Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #51cf66 0%, #40c057 100%); 
                                    padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                            <h2 style="margin: 0; font-size: 2rem;">✅ REAL NEWS</h2>
                            <p style="font-size: 1.2rem; margin: 1rem 0;">Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.markdown("### 📈 Confidence Scores")
                    classes = best_model['Model_Object'].classes_
                    prob_df = pd.DataFrame({
                        'Class': classes,
                        'Probability': probability
                    })
                    
                    fig = px.bar(prob_df, x='Class', y='Probability', 
                                color='Probability', color_continuous_scale='RdYlGn',
                                title="Class Probability Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🔍 Enter text and click 'Analyze Text' to get predictions")

# ================================================================
# 📝 Footer
# ================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔍 Fake News Detection System | Built with Streamlit & Machine Learning"
    "</div>",
    unsafe_allow_html=True
)
