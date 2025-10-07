# ================================================================
# 🧠 Fake News Detection App - Fixed for Small Datasets
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
from sklearn.utils import resample

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
    .warning-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
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
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
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
# 📊 Data Validation & Balancing Functions
# ================================================================
def validate_dataset(df, target_col):
    """Validate dataset for training suitability"""
    validation_results = {
        'is_valid': True,
        'issues': [],
        'class_distribution': None,
        'min_class_size': 0
    }
    
    # Check for missing values in target
    if df[target_col].isnull().any():
        validation_results['is_valid'] = False
        validation_results['issues'].append("Target column contains missing values")
    
    # Check class distribution
    class_dist = df[target_col].value_counts()
    validation_results['class_distribution'] = class_dist
    validation_results['min_class_size'] = class_dist.min()
    
    # Check if any class has less than 2 samples
    if class_dist.min() < 2:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"One or more classes have less than 2 samples. Class distribution: {dict(class_dist)}")
    
    # Check if we have at least 2 classes
    if len(class_dist) < 2:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Need at least 2 classes for classification")
    
    # Check text data quality
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_cols:
        empty_texts = df[col].apply(lambda x: len(str(x).strip()) == 0).sum()
        if empty_texts > 0:
            validation_results['issues'].append(f"Column '{col}' has {empty_texts} empty texts")
    
    return validation_results

def handle_imbalanced_data(df, text_col, target_col):
    """Handle imbalanced data with basic upsampling"""
    try:
        # Get class distribution
        class_dist = df[target_col].value_counts()
        max_size = class_dist.max()
        
        # Upsample minority classes
        balanced_dfs = []
        for class_name in class_dist.index:
            class_df = df[df[target_col] == class_name]
            if len(class_df) < max_size:
                # Upsample to match the majority class
                class_df_upsampled = resample(
                    class_df,
                    replace=True,
                    n_samples=max_size,
                    random_state=42
                )
                balanced_dfs.append(class_df_upsampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs)
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    except Exception as e:
        st.warning(f"Could not balance data: {str(e)}. Using original data.")
        return df

# ================================================================
# 📊 Visualization Functions
# ================================================================
def create_wordcloud(texts, title):
    """Create word cloud visualization"""
    try:
        texts = [text for text in texts if len(str(text).strip()) > 0]
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
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

def safe_train_test_split(X, y, test_size=0.2, min_samples_per_class=2):
    """Safe train-test split that handles small datasets"""
    try:
        # Check if we can do stratified split
        class_counts = pd.Series(y).value_counts()
        can_stratify = all(count >= 2 for count in class_counts)
        
        if can_stratify and len(class_counts) >= 2:
            return train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=42, 
                stratify=y
            )
        else:
            # Simple split without stratification
            st.warning("Using simple train-test split (not stratified) due to small class sizes")
            return train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=42
            )
    except Exception as e:
        st.error(f"Error in train-test split: {str(e)}")
        # Fallback: use all data for training
        return X, X[:0], y, pd.Series([], dtype=y.dtype)

def train_and_evaluate_models(X_train, X_test, y_train, y_test, selected_models):
    """Train and evaluate multiple models with error handling"""
    results = []
    
    for name in selected_models:
        try:
            model = models[name]
            model.fit(X_train, y_train)
            
            # Only calculate metrics if we have test data
            if len(X_test) > 0 and len(y_test) > 0:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Model_Object': model,
                    'Predictions': y_pred,
                    'Probabilities': model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                })
            else:
                # No test data - just store the model
                results.append({
                    'Model': name,
                    'Accuracy': 0.0,
                    'Model_Object': model,
                    'Predictions': [],
                    'Probabilities': None
                })
                st.warning(f"Model '{name}' trained but not evaluated (no test data)")
            
        except Exception as e:
            st.warning(f"❌ Failed to train {name}: {str(e)}")
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# ================================================================
# 🚀 Streamlit App Layout
# ================================================================
st.markdown("<h1 class='main-header'>🔍 Advanced Fake News Detection System</h1>", unsafe_allow_html=True)

# Initialize session state
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
    
    st.markdown("### Model Settings")
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    vectorizer_type = st.selectbox("Vectorization Method", ["TF-IDF", "Count Vectorizer"])
    max_features = st.slider("Max Features", 100, 5000, 1000)
    
    st.markdown("### Data Handling")
    balance_data = st.checkbox("Balance classes (upsample minority)", value=True)
    handle_small_classes = st.checkbox("Handle small classes automatically", value=True)
    
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
                    
                    # Validate dataset
                    validation = validate_dataset(df, target_col)
                    
                    if validation['is_valid']:
                        st.success("✅ Dataset loaded and validated successfully!")
                    else:
                        st.warning("⚠️ Dataset has issues:")
                        for issue in validation['issues']:
                            st.error(issue)
                        
                        st.info("💡 Tips to fix:")
                        st.write("- Ensure each class has at least 2 samples")
                        st.write("- Remove rows with missing target values")
                        st.write("- Check for empty text fields")
                    
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
                class_dist = df[target_col].value_counts()
                min_class = class_dist.min()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Min Class Size</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: {'#e74c3c' if min_class < 2 else '#27ae60'};">{min_class}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                empty_texts = df[text_col].apply(lambda x: len(str(x).strip()) == 0).sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝 Empty Texts</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: {'#e74c3c' if empty_texts > 0 else '#27ae60'};">{empty_texts}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Class distribution
            st.markdown("### 📊 Class Distribution")
            fig = plot_class_distribution(df[target_col])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

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
                        
                        # Handle class imbalance if requested
                        if balance_data:
                            processed_df = handle_imbalanced_data(processed_df, 'cleaned_text', target_col)
                            st.info("✅ Data balanced using upsampling")
                        
                        st.session_state.processed_data = processed_df
                        
                        # Initialize vectorizer
                        if vectorizer_type == "TF-IDF":
                            st.session_state.vectorizer = TfidfVectorizer(
                                max_features=max_features, 
                                ngram_range=(1, 2),
                                min_df=2  # Ignore terms that appear in less than 2 documents
                            )
                        else:
                            st.session_state.vectorizer = CountVectorizer(
                                max_features=max_features, 
                                ngram_range=(1, 2),
                                min_df=2
                            )
                        
                        st.success("✅ Preprocessing completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Preprocessing failed: {str(e)}")
        
        with col2:
            if st.session_state.processed_data is not None:
                processed_df = st.session_state.processed_data
                
                # Show preprocessing results
                st.markdown("### 📋 Preprocessing Results")
                
                tab1, tab2, tab3 = st.tabs(["Text Samples", "Statistics", "Class Distribution"])
                
                with tab1:
                    comparison_df = pd.DataFrame({
                        'Original Text': df[text_col].head(5),
                        'Cleaned Text': processed_df['cleaned_text'].head(5)
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                
                with tab2:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_length = processed_df['text_length'].mean()
                        st.metric("Avg Text Length", f"{avg_length:.1f} chars")
                    with col2:
                        avg_words = processed_df['word_count'].mean()
                        st.metric("Avg Word Count", f"{avg_words:.1f} words")
                    with col3:
                        st.metric("Total Samples", len(processed_df))
                
                with tab3:
                    fig = plot_class_distribution(processed_df[target_col])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# 🤖 SECTION 4: Model Training (Fixed)
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
        
        # Data validation before training
        validation = validate_dataset(processed_df, target_col)
        
        if not validation['is_valid']:
            st.error("❌ Dataset not suitable for training:")
            for issue in validation['issues']:
                st.error(issue)
            
            st.markdown("""
            <div class="warning-card">
                <h3>💡 How to fix:</h3>
                <ul>
                    <li>Ensure each class has at least 2 samples</li>
                    <li>Remove rows with missing target values</li>
                    <li>Enable 'Balance classes' in preprocessing</li>
                    <li>Check your dataset for sufficient data</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### ⚙️ Training Configuration")
                
                if st.button("🎯 Train Models", type="primary", use_container_width=True):
                    with st.spinner("Training models... This may take a few minutes."):
                        try:
                            # Prepare features and labels
                            X = vectorizer.fit_transform(processed_df['cleaned_text'])
                            y = processed_df[target_col]
                            
                            # Safe train-test split
                            X_train, X_test, y_train, y_test = safe_train_test_split(
                                X, y, test_size=test_size/100
                            )
                            
                            # Train models
                            results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, selected_models)
                            
                            if not results_df.empty:
                                st.session_state.trained_models = results_df
                                st.session_state.X_test = X_test
                                st.session_state.y_test = y_test
                                st.success(f"✅ Successfully trained {len(results_df)} models!")
                            else:
                                st.error("❌ No models were successfully trained.")
                                
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
            
            with col2:
                if (st.session_state.trained_models is not None and 
                    not st.session_state.trained_models.empty):
                    
                    results_df = st.session_state.trained_models
                    
                    # Display results
                    st.markdown("### 📊 Model Performance")
                    
                    if len(results_df) > 0:
                        best_model = results_df.iloc[0]
                        
                        # Best model card
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🏆 Best Performing Model</h3>
                            <p style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{best_model['Model']}</p>
                            <p style="font-size: 1.2rem;">Accuracy: <strong>{best_model['Accuracy']:.2%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Model comparison
                        if len(results_df) > 1:
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
                        
                        # Detailed evaluation (only if we have test data)
                        if len(st.session_state.y_test) > 0:
                            st.markdown("### 📈 Detailed Evaluation")
                            y_pred = best_model['Predictions']
                            y_test = st.session_state.y_test
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
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
                                report = classification_report(y_test, y_pred, output_dict=True)
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
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
        
        if len(results_df) == 0:
            st.error("❌ No trained models available.")
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
                            cleaned_text = clean_text(user_text)
                            X_input = vectorizer.transform([cleaned_text])
                            
                            prediction = best_model['Model_Object'].predict(X_input)[0]
                            probability = best_model['Model_Object'].predict_proba(X_input)[0]
                            
                            st.session_state.prediction = prediction
                            st.session_state.probability = probability
                            st.session_state.cleaned_text = cleaned_text
                            
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
            
            with col2:
                if 'prediction' in st.session_state:
                    prediction = st.session_state.prediction
                    probability = st.session_state.probability
                    
                    st.markdown("### 📊 Prediction Results")
                    
                    confidence = max(probability)
                    predicted_class = prediction
                    
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
