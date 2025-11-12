import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import time
import warnings
warnings.filterwarnings('ignore')

# Initialize NLTK with error handling
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

initialize_nltk()

# Set page configuration
st.set_page_config(
    page_title="TruthDetector AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .humor-section {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        font-style: italic;
    }
    .fact-check-card {
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #b3d9ff;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PolitifactScraper:
    def __init__(self):
        self.base_url = "https://www.politifact.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_facts(self, start_date, end_date, max_pages=3):
        """Scrape Politifact data within date range - DEMO VERSION"""
        st.warning("🔒 For demonstration purposes, using sample data. In production, this would scrape real Politifact data.")
        
        # Return comprehensive sample data
        return self.get_sample_data()
    
    def get_sample_data(self):
        """Generate comprehensive sample data for demonstration"""
        sample_statements = [
            # True statements
            {'statement': 'The Earth orbits around the Sun in approximately 365 days.', 'rating': 'true', 'date': datetime(2024, 1, 15)},
            {'statement': 'Regular exercise has been proven to improve cardiovascular health and mental wellbeing.', 'rating': 'true', 'date': datetime(2024, 1, 14)},
            {'statement': 'Vaccines have significantly reduced the incidence of infectious diseases worldwide.', 'rating': 'mostly-true', 'date': datetime(2024, 1, 13)},
            {'statement': 'Climate change is causing global temperatures to rise at an unprecedented rate.', 'rating': 'true', 'date': datetime(2024, 1, 12)},
            {'statement': 'The Great Barrier Reef is the worlds largest coral reef system.', 'rating': 'true', 'date': datetime(2024, 1, 11)},
            
            # False statements
            {'statement': 'The Moon is made entirely of green cheese and is inhabited by mice.', 'rating': 'false', 'date': datetime(2024, 1, 10)},
            {'statement': 'Drinking bleach can cure COVID-19 and other viral infections.', 'rating': 'false', 'date': datetime(2024, 1, 9)},
            {'statement': 'The Earth is flat and surrounded by an ice wall that prevents us from falling off.', 'rating': 'pants-fire', 'date': datetime(2024, 1, 8)},
            {'statement': '5G networks spread coronavirus through radio waves and mind control.', 'rating': 'false', 'date': datetime(2024, 1, 7)},
            {'statement': 'Humans only use 10% of their brains capacity according to scientific studies.', 'rating': 'false', 'date': datetime(2024, 1, 6)},
            
            # Mixed statements
            {'statement': 'Chocolate consumption helps with weight loss when combined with specific diets.', 'rating': 'half-true', 'date': datetime(2024, 1, 5)},
            {'statement': 'Reading in dim light permanently damages your eyesight over time.', 'rating': 'mostly-false', 'date': datetime(2024, 1, 4)},
            {'statement': 'Shark attacks are more common than deaths from falling coconuts.', 'rating': 'true', 'date': datetime(2024, 1, 3)},
            {'statement': 'You need to drink eight glasses of water per day for optimal health.', 'rating': 'mostly-true', 'date': datetime(2024, 1, 2)},
            {'statement': 'Cracking your knuckles leads to arthritis in later life.', 'rating': 'false', 'date': datetime(2024, 1, 1)},
        ]
        
        return sample_statements

class NLPAnalyzer:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def preprocess_text(self, text):
        """Basic text preprocessing with robust error handling"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            # Simple tokenization (fallback if NLTK fails)
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            # Remove stopwords
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            
            return ' '.join(tokens)
        except Exception as e:
            st.error(f"Text preprocessing error: {str(e)}")
            return text.lower()  # Fallback to simple lowercase
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob with error handling"""
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        except:
            return 0.0  # Neutral sentiment as fallback
    
    def extract_features(self, text):
        """Extract basic text features with error handling"""
        try:
            words = text.split()
            return {
                'word_count': len(words),
                'char_count': len(text),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'sentiment': self.analyze_sentiment(text)
            }
        except:
            return {
                'word_count': 0,
                'char_count': 0,
                'avg_word_length': 0,
                'sentiment': 0.0
            }

class TruthDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(random_state=42, probability=True, kernel='linear')
        }
        self.results = {}
    
    def prepare_data(self, data):
        """Prepare data for training with robust error handling"""
        try:
            df = pd.DataFrame(data)
            
            # Preprocess text
            analyzer = NLPAnalyzer()
            df['processed_text'] = df['statement'].apply(analyzer.preprocess_text)
            
            # Convert ratings to binary (True vs False)
            true_ratings = ['true', 'mostly-true', 'half-true']
            df['is_true'] = df['rating'].apply(
                lambda x: 1 if any(true in str(x).lower() for true in true_ratings) else 0
            )
            
            # Ensure we have enough data
            if len(df) < 5:
                st.error("Insufficient data for training. Please add more samples.")
                return None, None, None
            
            # Vectorize text
            try:
                X_text = self.vectorizer.fit_transform(df['processed_text'])
            except:
                # Fallback: use simple count vectorizer
                from sklearn.feature_extraction.text import CountVectorizer
                self.vectorizer = CountVectorizer(max_features=100)
                X_text = self.vectorizer.fit_transform(df['processed_text'])
            
            # Add additional features
            features = []
            for text in df['statement']:
                feat = analyzer.extract_features(text)
                features.append([feat['word_count'], feat['char_count'], 
                               feat['avg_word_length'], feat['sentiment']])
            
            X_features = np.array(features)
            
            # Handle case where vectorization might fail
            if X_text.shape[1] == 0:
                X_combined = X_features
            else:
                X_combined = np.hstack([X_text.toarray(), X_features])
            
            y = df['is_true']
            
            return X_combined, y, df
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None, None, None
    
    def train_models(self, X, y):
        """Train all models and store results with error handling"""
        if X is None or y is None:
            return {}
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    self.results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'y_test': y_test,
                        'y_pred': y_pred
                    }
                    
                except Exception as model_error:
                    st.warning(f"Model {name} failed to train: {str(model_error)}")
                    continue
            
            return self.results
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return {}

def generate_humorous_critique(statement, prediction, confidence):
    """Generate humorous critique based on prediction"""
    
    jokes = {
        'true': [
            "This statement is so true, even my algorithm blushed! 🤖",
            "Truth detected! More reliable than my morning coffee. ☕",
            "This fact is solid - like grandma's cooking! 🍲",
            "Verified! Even the fact-checker's fact-checker approves. ✅",
            "So true, it made my binary heart skip a beat! 01010111💖"
        ],
        'false': [
            "This statement is stretching the truth more than yoga pants! 🧘",
            "False alert! This claim has more holes than Swiss cheese! 🧀",
            "This 'fact' is about as accurate as a weather forecast! ⛈️",
            "Warning: This statement may cause spontaneous eyebrow raises! 🤨",
            "So false, even my fake-news detector is offended! 🚨"
        ]
    }
    
    category = 'true' if prediction == 1 else 'false'
    joke = np.random.choice(jokes[category])
    
    confidence_comment = ""
    if confidence > 0.8:
        confidence_comment = " (I'm more confident about this than my Wi-Fi password!)"
    elif confidence < 0.6:
        confidence_comment = " (Take this with a grain of salt... and maybe some tequila!)"
    
    return f"**Humorous Verdict:** {joke}{confidence_comment}"

def create_visualizations(results, data):
    """Create performance visualizations with error handling"""
    
    try:
        # Model accuracy comparison
        if results:
            fig1 = go.Figure(data=[
                go.Bar(name='Accuracy', 
                       x=list(results.keys()), 
                       y=[result['accuracy'] for result in results.values()],
                       marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ])
            fig1.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Models',
                yaxis_title='Accuracy',
                template='plotly_white'
            )
        else:
            fig1 = go.Figure()
            fig1.update_layout(title='No model results available')
        
        # Rating distribution
        if isinstance(data, pd.DataFrame) and 'rating' in data.columns:
            rating_counts = data['rating'].value_counts()
            fig2 = px.pie(values=rating_counts.values, 
                          names=rating_counts.index,
                          title='Fact Check Rating Distribution',
                          color_discrete_sequence=px.colors.sequential.Blues_r)
        else:
            fig2 = px.pie(values=[1], names=['Sample'], title='Sample Data Distribution')
        
        # Confusion matrix for best model
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
            best_result = results[best_model_name]
            cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
            
            fig3 = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted False', 'Predicted True'],
                y=['Actual False', 'Actual True'],
                colorscale='Blues',
                showscale=True
            ))
            fig3.update_layout(title=f'Confusion Matrix - {best_model_name}')
        else:
            fig3 = go.Figure()
            fig3.update_layout(title='No confusion matrix available')
        
        return fig1, fig2, fig3
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        # Return empty figures
        return go.Figure(), go.Figure(), go.Figure()

def main():
    # Header
    st.markdown('<div class="main-header">🔍 TruthDetector AI</div>', unsafe_allow_html=True)
    st.markdown("### Advanced Fact-Checking with NLP and Machine Learning")
    
    # Initialize session state
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_section = st.sidebar.radio("Go to:", 
                                   ["Data Collection", "NLP Analysis", 
                                    "Model Performance", "Fact Checker", "About"])
    
    # Add disclaimer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ Disclaimer")
    st.sidebar.info(
        "This is a demonstration app for educational purposes. "
        "Always verify facts through multiple reliable sources."
    )
    
    if app_section == "Data Collection":
        st.markdown('<div class="sub-header">📊 Data Collection</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime(2024, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime(2024, 1, 15))
        
        if st.button("Load Sample Data", type="primary"):
            with st.spinner("Loading sample data..."):
                scraper = PolitifactScraper()
                sample_data = scraper.get_sample_data()
                
                if sample_data:
                    st.session_state.scraped_data = sample_data
                    st.success(f"Successfully loaded {len(sample_data)} sample fact checks!")
        
        if st.session_state.scraped_data is not None:
            st.subheader("Data Preview")
            
            # Convert to DataFrame for display
            df_display = pd.DataFrame(st.session_state.scraped_data)
            st.dataframe(df_display, use_container_width=True)
            
            # Data statistics
            st.subheader("Data Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Statements", len(st.session_state.scraped_data))
            with col2:
                true_count = len([s for s in st.session_state.scraped_data 
                                if any(r in s['rating'] for r in ['true', 'mostly-true', 'half-true'])])
                st.metric("True Statements", true_count)
            with col3:
                false_count = len([s for s in st.session_state.scraped_data 
                                 if 'false' in s['rating'] or 'pants-fire' in s['rating']])
                st.metric("False Statements", false_count)
            with col4:
                mixed_count = len(st.session_state.scraped_data) - true_count - false_count
                st.metric("Mixed Statements", mixed_count)
    
    elif app_section == "NLP Analysis":
        st.markdown('<div class="sub-header">🔤 Natural Language Processing Analysis</div>', unsafe_allow_html=True)
        
        if st.session_state.scraped_data is None:
            st.warning("Please load sample data first in the 'Data Collection' section.")
            if st.button("Load Sample Data Now"):
                scraper = PolitifactScraper()
                st.session_state.scraped_data = scraper.get_sample_data()
                st.rerun()
        else:
            analyzer = NLPAnalyzer()
            
            # Show NLP features
            st.subheader("Text Analysis Features")
            
            # Select statement to analyze
            statements = [item['statement'] for item in st.session_state.scraped_data]
            selected_index = st.selectbox("Select a statement to analyze:", 
                                         range(len(statements)),
                                         format_func=lambda x: statements[x][:100] + "..." if len(statements[x]) > 100 else statements[x])
            
            if selected_index is not None:
                sample_text = statements[selected_index]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Text:**")
                    st.info(sample_text)
                    
                    st.markdown("**Processed Text:**")
                    processed = analyzer.preprocess_text(sample_text)
                    st.success(processed)
                
                with col2:
                    features = analyzer.extract_features(sample_text)
                    st.markdown("**Text Features:**")
                    
                    # Create metrics for features
                    feat_col1, feat_col2 = st.columns(2)
                    with feat_col1:
                        st.metric("Word Count", features['word_count'])
                        st.metric("Character Count", features['char_count'])
                    with feat_col2:
                        st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
                        st.metric("Sentiment Score", f"{features['sentiment']:.2f}")
                    
                    # Sentiment interpretation
                    sentiment = features['sentiment']
                    if sentiment > 0.1:
                        sentiment_label = "😊 Positive"
                    elif sentiment < -0.1:
                        sentiment_label = "😠 Negative"
                    else:
                        sentiment_label = "😐 Neutral"
                    
                    st.write(f"**Sentiment:** {sentiment_label}")
    
    elif app_section == "Model Performance":
        st.markdown('<div class="sub-header">📈 Machine Learning Model Performance</div>', unsafe_allow_html=True)
        
        if st.session_state.scraped_data is None:
            st.warning("Please load sample data first in the 'Data Collection' section.")
        else:
            if st.button("Train Models", type="primary"):
                with st.spinner("Training machine learning models... This may take a few seconds."):
                    detector = TruthDetector()
                    X, y, processed_data = detector.prepare_data(st.session_state.scraped_data)
                    
                    if X is not None and y is not None:
                        results = detector.train_models(X, y)
                        
                        st.session_state.trained_models = detector
                        st.session_state.results = results
                        
                        if results:
                            st.success("Models trained successfully!")
                        else:
                            st.error("No models were successfully trained.")
                    else:
                        st.error("Could not prepare data for training.")
            
            if st.session_state.results:
                # Display results
                st.subheader("Model Performance Results")
                
                # Create visualizations
                fig1, fig2, fig3 = create_visualizations(
                    st.session_state.results, 
                    pd.DataFrame(st.session_state.scraped_data)
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Model comparison table
                st.subheader("Detailed Performance Metrics")
                performance_data = []
                for model_name, result in st.session_state.results.items():
                    performance_data.append({
                        'Model': model_name,
                        'Accuracy': f"{result['accuracy']:.3f}",
                        'Status': '✅ Trained'
                    })
                
                st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
                
                # Best model info
                best_model = max(st.session_state.results.items(), key=lambda x: x[1]['accuracy'])
                st.info(f"🎯 **Best Performing Model:** {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
    
    elif app_section == "Fact Checker":
        st.markdown('<div class="sub-header">🔍 Fact Checking Tool</div>', unsafe_allow_html=True)
        
        if st.session_state.trained_models is None:
            st.warning("Please train the models first in the 'Model Performance' section.")
            
            if st.button("Train Models Now"):
                with st.spinner("Training models..."):
                    detector = TruthDetector()
                    X, y, _ = detector.prepare_data(st.session_state.scraped_data)
                    detector.train_models(X, y)
                    st.session_state.trained_models = detector
                    st.session_state.results = detector.results
                st.rerun()
        else:
            st.markdown("### Check Statement Credibility")
            
            # User input
            user_statement = st.text_area(
                "Enter a statement to check:", 
                "The Earth revolves around the Sun.",
                height=100
            )
            
            selected_model = st.selectbox(
                "Select NLP Model for Analysis:",
                list(st.session_state.trained_models.models.keys())
            )
            
            if st.button("Analyze Credibility", type="primary"):
                if not user_statement.strip():
                    st.error("Please enter a statement to analyze.")
                else:
                    with st.spinner("Analyzing statement..."):
                        # Preprocess and predict
                        analyzer = NLPAnalyzer()
                        processed_text = analyzer.preprocess_text(user_statement)
                        
                        try:
                            # Vectorize text
                            X_text = st.session_state.trained_models.vectorizer.transform([processed_text])
                            
                            # Add features
                            features = analyzer.extract_features(user_statement)
                            X_features = np.array([[features['word_count'], features['char_count'], 
                                                  features['avg_word_length'], features['sentiment']]])
                            
                            X_combined = np.hstack([X_text.toarray(), X_features])
                            
                            # Predict
                            model = st.session_state.trained_models.results[selected_model]['model']
                            prediction = model.predict(X_combined)[0]
                            probability = model.predict_proba(X_combined)[0][prediction]
                            
                            # Display results
                            st.markdown('<div class="fact-check-card">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                verdict = "✅ LIKELY TRUE" if prediction == 1 else "❌ LIKELY FALSE"
                                color = "green" if prediction == 1 else "red"
                                st.markdown(f"### <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
                                st.metric("Confidence Score", f"{probability:.2%}")
                                st.write(f"**Model Used:** {selected_model}")
                            
                            with col2:
                                features = analyzer.extract_features(user_statement)
                                st.write("**Analysis Details:**")
                                st.write(f"- 📝 Word Count: {features['word_count']}")
                                st.write(f- "😊 Sentiment: {features['sentiment']:.2f}")
                                st.write(f"- 🔤 Characters: {features['char_count']}")
                                st.write(f"- 📏 Avg Word Length: {features['avg_word_length']:.1f}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Humorous critique
                            humorous_verdict = generate_humorous_critique(user_statement, prediction, probability)
                            st.markdown(f'<div class="humor-section">{humorous_verdict}</div>', unsafe_allow_html=True)
                            
                            # Credibility assessment
                            st.subheader("🔍 Credibility Assessment")
                            
                            credibility_score = probability if prediction == 1 else 1 - probability
                            
                            if credibility_score > 0.8:
                                assessment = "**High Credibility** - This statement appears reliable based on our analysis."
                                icon = "🟢"
                            elif credibility_score > 0.6:
                                assessment = "**Moderate Credibility** - This statement seems plausible but verify with additional sources."
                                icon = "🟡"
                            else:
                                assessment = "**Low Credibility** - Exercise caution and verify this statement with trusted sources."
                                icon = "🔴"
                            
                            st.info(f"{icon} {assessment}")
                            
                            # Progress bar for credibility score
                            st.write(f"**Overall Credibility Score:** {credibility_score:.2%}")
                            st.progress(float(credibility_score))
                            
                            # User guidance
                            st.markdown("---")
                            st.subheader("📋 Fact-Checking Guidance")
                            st.markdown("""
                            When evaluating statements, consider:
                            - **Source Reliability**: Where did this information originate?
                            - **Corroboration**: Do other reputable sources confirm this?
                            - **Evidence**: Is there scientific or verifiable evidence?
                            - **Logical Consistency**: Does this make logical sense?
                            - **Expert Consensus**: What do experts in the field say?
                            """)
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            st.info("Please try training the models again or use a different statement.")
    
    else:  # About section
        st.markdown('<div class="sub-header">ℹ️ About TruthDetector AI</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### How It Works
        
        **TruthDetector AI** uses advanced Natural Language Processing and Machine Learning to analyze statements and assess their credibility.
        
        ### Methodology
        
        1. **Data Collection**: Uses sample fact-checked statements (Politifact-style)
        2. **NLP Processing**: Analyzes text features, sentiment, and linguistic patterns
        3. **Machine Learning**: Trains multiple models to detect truth patterns
        4. **Credibility Assessment**: Provides confidence scores and insights
        
        ### Models Used
        
        - **Decision Tree**: Rule-based classification
        - **Logistic Regression**: Statistical probability modeling  
        - **Naive Bayes**: Probabilistic classification
        - **SVM**: Advanced pattern recognition
        
        ### Technical Features
        
        - **Text Preprocessing**: Tokenization, stopword removal, feature extraction
        - **Sentiment Analysis**: Emotional tone assessment using TextBlob
        - **Feature Engineering**: Word counts, character counts, average word length
        - **Model Evaluation**: Accuracy scores, confusion matrices, performance comparison
        
        ### Important Notes
        
        🔒 **This is a demonstration app** for educational purposes
        📊 **Sample data** is used instead of live scraping for reliability
        🤖 **AI limitations**: Models are trained on limited sample data
        🔍 **Always verify** important information through multiple reliable sources
        
        ### User Prompt for Fact Checking
        
        When evaluating information credibility, ask yourself:
        
        - ❓ **Source**: Who is sharing this information and what's their expertise?
        - ❓ **Evidence**: What verifiable evidence supports this claim?
        - ❓ **Consensus**: Do experts in the field generally agree on this?
        - ❓ **Logic**: Does this claim make logical sense?
        - ❓ **Bias**: Could there be any agenda or bias influencing this information?
        - ❓ **Corroboration**: Can this be verified through multiple independent sources?
        """)

if __name__ == "__main__":
    main()
