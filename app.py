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
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
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

initialize_nltk()

# Set page configuration
st.set_page_config(
    page_title="TruthDetector AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced professional UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-left: 5px solid #667eea;
        padding-left: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .humor-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        font-style: italic;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .fact-check-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: none;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: none;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #ffd43b;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: none;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: none;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #17a2b8;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateX(5px);
    }
    
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top: 4px solid;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

class PolitifactScraper:
    def __init__(self):
        self.base_url = "https://www.politifact.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_comprehensive_sample_data(self):
        """Generate comprehensive and realistic sample data"""
        sample_data = [
            {
                'statement': 'NASA confirmed that 2023 was the hottest year on record globally, with temperatures 1.4°C above pre-industrial levels.',
                'rating': 'true', 
                'date': datetime(2024, 1, 15),
                'category': 'Environment'
            },
            {
                'statement': 'Regular physical activity can reduce the risk of depression and anxiety by up to 30% according to WHO studies.',
                'rating': 'mostly-true', 
                'date': datetime(2024, 1, 14),
                'category': 'Health'
            },
            {
                'statement': 'The James Webb Space Telescope has discovered galaxies that formed less than 400 million years after the Big Bang.',
                'rating': 'true', 
                'date': datetime(2024, 1, 13),
                'category': 'Science'
            },
            {
                'statement': 'Drinking alkaline water can cure cancer and reverse aging by changing your body pH levels permanently.',
                'rating': 'false', 
                'date': datetime(2024, 1, 12),
                'category': 'Health'
            },
            {
                'statement': '5G towers are government surveillance devices that can control human thoughts through radio waves.',
                'rating': 'pants-fire', 
                'date': datetime(2024, 1, 11),
                'category': 'Technology'
            },
            {
                'statement': 'The COVID-19 vaccine contains microchips that allow Bill Gates to track people worldwide.',
                'rating': 'false', 
                'date': datetime(2024, 1, 10),
                'category': 'Health'
            },
            {
                'statement': 'Eating chocolate every day helps you lose weight by boosting metabolism significantly.',
                'rating': 'half-true', 
                'date': datetime(2024, 1, 9),
                'category': 'Health'
            },
            {
                'statement': 'Artificial intelligence will replace 50% of all jobs within the next two years.',
                'rating': 'mostly-false', 
                'date': datetime(2024, 1, 8),
                'category': 'Technology'
            },
            {
                'statement': 'Renewable energy sources now provide over 80% of global electricity demand.',
                'rating': 'false', 
                'date': datetime(2024, 1, 7),
                'category': 'Environment'
            },
            {
                'statement': 'The Great Pyramid of Giza was built by aliens using advanced technology not available to ancient Egyptians.',
                'rating': 'pants-fire', 
                'date': datetime(2024, 1, 6),
                'category': 'History'
            },
            {
                'statement': 'Meditation and mindfulness practices can reduce stress and improve focus in workplace settings.',
                'rating': 'true', 
                'date': datetime(2024, 1, 5),
                'category': 'Health'
            },
            {
                'statement': 'Electric vehicles produce more carbon emissions than gasoline cars when accounting for battery production.',
                'rating': 'mostly-false', 
                'date': datetime(2024, 1, 4),
                'category': 'Environment'
            }
        ]
        
        return sample_data

class NLPAnalyzer:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs, special characters, and extra spaces
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenization with fallback
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            # Remove stopwords and short tokens
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            
            return ' '.join(tokens)
        except Exception as e:
            return text.lower()
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis"""
        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'label': 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral'
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0, 'label': 'Neutral'}
    
    def extract_linguistic_features(self, text):
        """Extract comprehensive linguistic features"""
        try:
            words = text.split()
            sentences = [s for s in text.split('.') if s.strip()]
            
            # Calculate readability metrics (simplified)
            avg_sentence_length = len(words) / len(sentences) if len(sentences) > 0 else len(words)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Count specific word types
            long_words = [word for word in words if len(word) > 6]
            unique_words = set(words)
            
            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'char_count': len(text),
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'unique_word_ratio': len(unique_words) / len(words) if words else 0,
                'long_word_ratio': len(long_words) / len(words) if words else 0,
                'lexical_diversity': len(unique_words) / len(words) if words else 0
            }
        except:
            return {
                'word_count': 0, 'sentence_count': 0, 'char_count': 0,
                'avg_word_length': 0, 'avg_sentence_length': 0,
                'unique_word_ratio': 0, 'long_word_ratio': 0, 'lexical_diversity': 0
            }

class TruthDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'SVM': SVC(random_state=42, probability=True, kernel='linear', C=0.1)
        }
        self.results = {}
    
    def prepare_data(self, data):
        """Enhanced data preparation"""
        try:
            df = pd.DataFrame(data)
            
            # Enhanced preprocessing
            analyzer = NLPAnalyzer()
            df['processed_text'] = df['statement'].apply(analyzer.preprocess_text)
            
            # Convert ratings to binary (True vs False/Mixed)
            true_ratings = ['true', 'mostly-true']
            false_ratings = ['false', 'pants-fire', 'mostly-false']
            
            def map_rating(rating):
                if rating in true_ratings:
                    return 1  # True
                elif rating in false_ratings:
                    return 0  # False
                else:
                    return 1  # Consider half-true as true for binary classification
            
            df['is_true_binary'] = df['rating'].apply(map_rating)
            
            # Vectorize text
            X_text = self.vectorizer.fit_transform(df['processed_text'])
            
            # Extract comprehensive features
            features_list = []
            for text in df['statement']:
                linguistic_feat = analyzer.extract_linguistic_features(text)
                sentiment_feat = analyzer.analyze_sentiment(text)
                
                features = [
                    linguistic_feat['word_count'],
                    linguistic_feat['sentence_count'],
                    linguistic_feat['char_count'],
                    linguistic_feat['avg_word_length'],
                    linguistic_feat['avg_sentence_length'],
                    linguistic_feat['unique_word_ratio'],
                    linguistic_feat['long_word_ratio'],
                    linguistic_feat['lexical_diversity'],
                    sentiment_feat['polarity'],
                    sentiment_feat['subjectivity']
                ]
                features_list.append(features)
            
            X_features = np.array(features_list)
            X_combined = np.hstack([X_text.toarray(), X_features])
            
            return X_combined, df['is_true_binary'], df
            
        except Exception as e:
            st.error(f"Data preparation error: {str(e)}")
            return None, None, None
    
    def train_models(self, X, y):
        """Enhanced model training with performance tracking"""
        if X is None or y is None:
            return {}
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            
            for name, model in self.models.items():
                try:
                    # Train model
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Store results
                    self.results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'training_time': training_time,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'confusion_matrix': cm
                    }
                    
                except Exception as model_error:
                    continue
            
            return self.results
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return {}

def generate_humorous_critique(statement, prediction, confidence, features):
    """Generate enhanced humorous critiques with personality"""
    
    truth_jokes = {
        'high_confidence_true': [
            "🎯 Bullseye! This statement is so true, even my algorithms are impressed!",
            "✅ Verified! More reliable than your morning alarm clock!",
            "🏆 Truth champion! This fact could win awards for accuracy!",
            "🌟 Stellar accuracy! Even the fact-checker's fact-checker approves!",
            "💎 Diamond-grade truth! This statement is polished to perfection!"
        ],
        'medium_confidence_true': [
            "📗 Likely true! I'd bet my virtual coffee on this one!",
            "👍 Looking good! This statement passes the sniff test!",
            "🔍 Promising! More reliable than weather forecasts!",
            "📚 Educated guess: This seems legit!",
            "🎲 Probably true! Better odds than lottery tickets!"
        ],
        'high_confidence_false': [
            "🚨 False alarm! This statement has more red flags than a matador convention!",
            "🎭 Pure fiction! More made-up than my excuses for being late!",
            "🧀 Full of holes! Swiss cheese is more solid than this claim!",
            "🌈 Unicorn territory! This is more fantasy than reality!",
            "🎪 Circus act! This claim is juggling too many falsehoods!"
        ],
        'medium_confidence_false': [
            "🤔 Suspicious! This smells fishier than a seafood market!",
            "⚠️ Dubious claim! I'm getting 'alternative facts' vibes!",
            "🎈 Inflated truth! This balloon is about to pop!",
            "🕵️ Investigate further! My truth-o-meter is twitching!",
            "📉 Questionable! This claim is on shaky ground!"
        ]
    }
    
    # Determine joke category
    if prediction == 1:
        category = 'high_confidence_true' if confidence > 0.8 else 'medium_confidence_true'
    else:
        category = 'high_confidence_false' if confidence > 0.8 else 'medium_confidence_false'
    
    joke = np.random.choice(truth_jokes[category])
    
    # Add feature-based humor
    word_count = features.get('word_count', 0)
    if word_count > 50:
        joke += " And it's quite the mouthful! 🗣️"
    elif word_count < 10:
        joke += " Short and... questionable! 📝"
    
    sentiment = features.get('sentiment', {}).get('polarity', 0)
    if sentiment > 0.3:
        joke += " Positively misleading! 😊"
    elif sentiment < -0.3:
        joke += " Negatively charged! ⚡"
    
    return joke

def create_feature_analysis_visualization(analyzer, text):
    """Create feature analysis visualization for a given text"""
    features = analyzer.extract_linguistic_features(text)
    sentiment = analyzer.analyze_sentiment(text)
    
    # Create feature radar chart
    categories = ['Word Count', 'Sentence Count', 'Avg Word Length', 
                 'Unique Words', 'Long Words', 'Lexical Diversity']
    
    values = [
        min(features['word_count'] / 100, 1),  # Normalize
        min(features['sentence_count'] / 10, 1),
        min(features['avg_word_length'] / 10, 1),
        features['unique_word_ratio'],
        features['long_word_ratio'],
        features['lexical_diversity']
    ]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea'),
        name='Text Features'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title='Linguistic Feature Analysis'
    )
    
    return fig_radar, features, sentiment

def create_enhanced_visualizations(results, data):
    """Create stunning visualizations with Plotly"""
    
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # 1. Model Performance Comparison
        if results:
            models = list(results.keys())
            accuracies = [results[model]['accuracy'] for model in models]
            
            fig_comparison = go.Figure(data=[
                go.Bar(
                    name='Accuracy', 
                    x=models, 
                    y=accuracies,
                    marker_color=['#667eea', '#4ecdc4', '#ff6b6b', '#ffd93d'],
                    text=[f'{acc:.3f}' for acc in accuracies],
                    textposition='auto',
                )
            ])
            
            fig_comparison.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Models',
                yaxis_title='Accuracy',
                template='plotly_white',
                showlegend=False
            )
        else:
            fig_comparison = go.Figure()
            fig_comparison.update_layout(title='No model results available')
        
        # 2. Enhanced Rating Distribution with Donut Chart
        rating_counts = df['rating'].value_counts()
        colors = px.colors.qualitative.Set3
        
        fig_donut = px.pie(
            values=rating_counts.values,
            names=rating_counts.index,
            title='Fact Check Rating Distribution',
            hole=0.4,
            color_discrete_sequence=colors
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(showlegend=False)
        
        # 3. Category Distribution
        category_counts = df['category'].value_counts()
        fig_sunburst = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Statement Categories',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        # 4. Best Model Confusion Matrix
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
            best_result = results[best_model_name]
            cm = best_result['confusion_matrix']
            
            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=['Predicted False', 'Predicted True'],
                y=['Actual False', 'Actual True'],
                colorscale='Blues',
                showscale=True
            )
            fig_cm.update_layout(title=f'Confusion Matrix - {best_model_name}')
        else:
            fig_cm = go.Figure()
            fig_cm.update_layout(title='No confusion matrix available')
        
        # 5. Simple radar chart placeholder
        fig_radar = go.Figure()
        fig_radar.update_layout(title='Performance Radar (Placeholder)')
        
        return fig_radar, fig_donut, fig_comparison, fig_sunburst, fig_cm
        
    except Exception as e:
        # Return empty figures on error
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

# Initialize session state
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = None
if 'results' not in st.session_state:
    st.session_state.results = None

def show_dashboard():
    """Enhanced Dashboard View"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 3.5rem; margin: 0;'>🔍 TruthDetector AI</h1>
        <p style='color: white; font-size: 1.2rem; opacity: 0.9;'>Advanced Fact-Checking with NLP & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class='info-box'>
        <h3>🚀 Welcome to TruthDetector AI</h3>
        <p>Your comprehensive solution for automated fact-checking using cutting-edge 
        Natural Language Processing and Machine Learning technologies.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
        <h4>🎯 Key Features</h4>
        <ul>
        <li>Advanced NLP Analysis</li>
        <li>Multiple ML Models</li>
        <li>Real-time Fact Checking</li>
        <li>Visual Analytics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start cards
    st.markdown("### 🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='model-card' style='border-top-color: #667eea;'>
        <h4>📊 Load Data</h4>
        <p>Start by loading sample fact-checked data to train your models.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='model-card' style='border-top-color: #ff6b6b;'>
        <h4>🤖 Train Models</h4>
        <p>Train multiple machine learning models on the loaded data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='model-card' style='border-top-color: #4ecdc4;'>
        <h4>🔍 Check Facts</h4>
        <p>Use trained models to analyze new statements for credibility.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample visualization placeholder
    scraper = PolitifactScraper()
    sample_data = scraper.get_comprehensive_sample_data()
    df = pd.DataFrame(sample_data)
    
    st.markdown("### 📈 Sample Insights")
    col1, col2 = st.columns(2)
    with col1:
        rating_counts = df['rating'].value_counts()
        fig = px.pie(values=rating_counts.values, names=rating_counts.index, 
                    title="Sample Data Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        category_counts = df['category'].value_counts()
        fig = px.bar(x=category_counts.index, y=category_counts.values,
                    title="Statements by Category", color=category_counts.index)
        st.plotly_chart(fig, use_container_width=True)

def show_data_collection():
    """Enhanced Data Collection Section"""
    st.markdown('<div class="sub-header">📊 Data Collection</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h4>📝 About the Data</h4>
    <p>This app uses carefully curated sample data that mimics real fact-checked statements from various categories 
    including Science, Health, Technology, and Environment. The data includes both true and false statements 
    with different confidence levels.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Load Sample Data", type="primary", use_container_width=True):
        with st.spinner("Loading comprehensive sample data..."):
            scraper = PolitifactScraper()
            sample_data = scraper.get_comprehensive_sample_data()
            
            if sample_data:
                st.session_state.scraped_data = sample_data
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success(f"✅ Successfully loaded {len(sample_data)} sample fact checks!")
                st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.scraped_data:
        # Enhanced data display
        st.markdown("### 📋 Data Preview")
        
        # Convert to DataFrame for better display
        df_display = pd.DataFrame(st.session_state.scraped_data)
        
        # Style the DataFrame
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400
        )
        
        # Enhanced statistics with metrics
        st.markdown("### 📊 Data Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = len(st.session_state.scraped_data)
            st.markdown(f"""
            <div class='metric-card'>
                <h3>📝 Total</h3>
                <h2>{total}</h2>
                <p>Statements</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            true_count = len([s for s in st.session_state.scraped_data 
                            if s['rating'] in ['true', 'mostly-true']])
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);'>
                <h3>✅ True</h3>
                <h2>{true_count}</h2>
                <p>Verified Facts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            false_count = len([s for s in st.session_state.scraped_data 
                             if s['rating'] in ['false', 'pants-fire', 'mostly-false']])
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);'>
                <h3>❌ False</h3>
                <h2>{false_count}</h2>
                <p>Debunked Claims</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            mixed_count = len([s for s in st.session_state.scraped_data 
                             if s['rating'] in ['half-true']])
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #ffd93d 0%, #ff9c3d 100%);'>
                <h3>⚖️ Mixed</h3>
                <h2>{mixed_count}</h2>
                <p>Partial Truths</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📈 Data Overview")
        df = pd.DataFrame(st.session_state.scraped_data)
        
        col1, col2 = st.columns(2)
        with col1:
            # Rating distribution
            rating_counts = df['rating'].value_counts()
            fig = px.pie(
                values=rating_counts.values, 
                names=rating_counts.index,
                title="📊 Rating Distribution",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category distribution
            category_counts = df['category'].value_counts()
            fig = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                title="📚 Categories Distribution",
                color=category_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_nlp_analysis():
    """Enhanced NLP Analysis Section"""
    st.markdown('<div class="sub-header">🔤 Natural Language Processing Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.scraped_data is None:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("Please load sample data first in the 'Data Collection' section.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("📥 Load Sample Data Now"):
            scraper = PolitifactScraper()
            st.session_state.scraped_data = scraper.get_comprehensive_sample_data()
            st.rerun()
    else:
        analyzer = NLPAnalyzer()
        
        # Enhanced statement selection
        st.markdown("### 🔍 Text Analysis Explorer")
        
        statements = [item['statement'] for item in st.session_state.scraped_data]
        selected_index = st.selectbox(
            "Select a statement to analyze:",
            range(len(statements)),
            format_func=lambda x: f"{statements[x][:80]}..." if len(statements[x]) > 80 else statements[x]
        )
        
        if selected_index is not None:
            sample_text = statements[selected_index]
            selected_statement = st.session_state.scraped_data[selected_index]
            
            # Create two main columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Original text display
                st.markdown("#### 📝 Original Statement")
                st.markdown(f'<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">{sample_text}</div>', 
                           unsafe_allow_html=True)
                
                # Rating information
                rating = selected_statement['rating']
                rating_color = {
                    'true': '#28a745',
                    'mostly-true': '#20c997',
                    'half-true': '#ffc107',
                    'mostly-false': '#fd7e14',
                    'false': '#dc3545',
                    'pants-fire': '#dc3545'
                }.get(rating, '#6c757d')
                
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; margin: 1rem 0;'>
                    <span><strong>Rating:</strong></span>
                    <span style='color: {rating_color}; font-weight: bold; text-transform: uppercase;'>{rating}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Processed text
                st.markdown("#### 🔧 Processed Text")
                processed = analyzer.preprocess_text(sample_text)
                st.markdown(f'<div style="background: #e7f3ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #17a2b8;">{processed}</div>', 
                           unsafe_allow_html=True)
            
            with col2:
                # Feature analysis visualization
                st.markdown("#### 📊 Feature Analysis")
                fig_radar, features, sentiment = create_feature_analysis_visualization(analyzer, sample_text)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Detailed features in columns
            st.markdown("#### 📈 Detailed Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### 📝 Linguistic Features")
                st.metric("Word Count", features['word_count'])
                st.metric("Sentence Count", features['sentence_count'])
                st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
                st.metric("Character Count", features['char_count'])
            
            with col2:
                st.markdown("##### 🎯 Complexity Metrics")
                st.metric("Unique Word Ratio", f"{features['unique_word_ratio']:.2%}")
                st.metric("Long Word Ratio", f"{features['long_word_ratio']:.2%}")
                st.metric("Lexical Diversity", f"{features['lexical_diversity']:.2f}")
                st.metric("Avg Sentence Length", f"{features['avg_sentence_length']:.1f}")
            
            with col3:
                st.markdown("##### 😊 Sentiment Analysis")
                
                # Sentiment gauge
                sentiment_value = sentiment['polarity']
                sentiment_color = "green" if sentiment_value > 0.1 else "red" if sentiment_value < -0.1 else "orange"
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sentiment_value,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Polarity"},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': sentiment_color},
                        'steps': [
                            {'range': [-1, -0.1], 'color': "lightgray"},
                            {'range': [-0.1, 0.1], 'color': "lightyellow"},
                            {'range': [0.1, 1], 'color': "lightgreen"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.metric("Subjectivity", f"{sentiment['subjectivity']:.2f}")
                st.metric("Sentiment Label", sentiment['label'])

def show_model_performance():
    """Enhanced Model Performance Section"""
    st.markdown('<div class="sub-header">📈 Machine Learning Model Performance</div>', unsafe_allow_html=True)
    
    if st.session_state.scraped_data is None:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("Please load sample data first in the 'Data Collection' section.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("📥 Load Sample Data Now"):
            scraper = PolitifactScraper()
            st.session_state.scraped_data = scraper.get_comprehensive_sample_data()
            st.rerun()
    else:
        # Training section
        st.markdown("### 🤖 Model Training")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            <div class='info-box'>
            Train multiple machine learning models to detect truth patterns in statements. 
            Each model uses different algorithms to provide diverse perspectives on fact-checking.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Train All Models", type="primary", use_container_width=True):
                with st.spinner("🧠 Training machine learning models... This may take a few seconds."):
                    detector = TruthDetector()
                    X, y, processed_data = detector.prepare_data(st.session_state.scraped_data)
                    
                    if X is not None and y is not None:
                        results = detector.train_models(X, y)
                        
                        st.session_state.trained_models = detector
                        st.session_state.results = results
                        
                        if results:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success(f"✅ Successfully trained {len(results)} models!")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("❌ No models were successfully trained.")
                    else:
                        st.error("❌ Could not prepare data for training.")
        
        if st.session_state.results:
            # Enhanced results display
            st.markdown("### 📊 Performance Results")
            
            # Create all visualizations
            fig_radar, fig_donut, fig_comparison, fig_sunburst, fig_cm = create_enhanced_visualizations(
                st.session_state.results, 
                st.session_state.scraped_data
            )
            
            # Model cards with performance metrics
            st.markdown("#### 🎯 Model Performance Cards")
            
            cols = st.columns(4)
            model_colors = {
                'Decision Tree': '#4ecdc4',
                'Logistic Regression': '#667eea', 
                'Naive Bayes': '#ff6b6b',
                'SVM': '#ffd93d'
            }
            
            for idx, (model_name, result) in enumerate(st.session_state.results.items()):
                with cols[idx % 4]:
                    accuracy = result['accuracy']
                    training_time = result['training_time']
                    
                    # Determine performance level
                    if accuracy > 0.8:
                        performance_emoji = "🏆"
                        performance_text = "Excellent"
                    elif accuracy > 0.7:
                        performance_emoji = "⭐"
                        performance_text = "Good"
                    elif accuracy > 0.6:
                        performance_emoji = "📈"
                        performance_text = "Fair"
                    else:
                        performance_emoji = "🔧"
                        performance_text = "Needs Improvement"
                    
                    st.markdown(f"""
                    <div class='model-card' style='border-top-color: {model_colors.get(model_name, '#667eea')};'>
                        <h4>{model_name}</h4>
                        <div style='font-size: 2rem; text-align: center; margin: 1rem 0;'>{performance_emoji}</div>
                        <div style='text-align: center;'>
                            <h3 style='color: {model_colors.get(model_name, '#667eea')}; margin: 0;'>{accuracy:.3f}</h3>
                            <p style='margin: 0; font-size: 0.9rem;'>Accuracy</p>
                        </div>
                        <div style='margin-top: 1rem;'>
                            <small>⏱️ {training_time:.2f}s</small><br>
                            <small>{performance_text}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Main visualizations
            st.markdown("#### 📈 Performance Visualizations")
            
            # Row 1: Comparison and Donut
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_comparison, use_container_width=True)
            with col2:
                st.plotly_chart(fig_donut, use_container_width=True)
            
            # Row 2: Data distribution and confusion matrix
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_sunburst, use_container_width=True)
            with col2:
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # Best model highlight
            best_model_name = max(st.session_state.results.items(), key=lambda x: x[1]['accuracy'])[0]
            best_accuracy = st.session_state.results[best_model_name]['accuracy']
            
            st.markdown(f"""
            <div class='success-box'>
                <h3>🎯 Best Performing Model: {best_model_name}</h3>
                <p>With an accuracy of <strong>{best_accuracy:.3f}</strong>, this model provides the most reliable predictions.</p>
            </div>
            """, unsafe_allow_html=True)

def show_fact_checker():
    """Enhanced Fact Checker Section"""
    st.markdown('<div class="sub-header">🔍 Fact Checking Tool</div>', unsafe_allow_html=True)
    
    if st.session_state.trained_models is None:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("Please train the models first in the 'Model Performance' section.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🤖 Train Models Now"):
            with st.spinner("Training models with sample data..."):
                detector = TruthDetector()
                X, y, _ = detector.prepare_data(st.session_state.scraped_data)
                detector.train_models(X, y)
                st.session_state.trained_models = detector
                st.session_state.results = detector.results
            st.rerun()
    else:
        # Main fact checking interface
        st.markdown("### 🎯 Check Statement Credibility")
        
        # User input section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_statement = st.text_area(
                "Enter a statement to analyze:",
                "The Earth revolves around the Sun in approximately 365.25 days.",
                height=120,
                placeholder="Type or paste the statement you want to fact-check here..."
            )
        
        with col2:
            st.markdown("#### ⚙️ Settings")
            selected_model = st.selectbox(
                "Analysis Model:",
                list(st.session_state.trained_models.models.keys()),
                index=1  # Default to Logistic Regression
            )
        
        # Analyze button
        if st.button("🔍 Analyze Credibility", type="primary", use_container_width=True):
            if not user_statement.strip():
                st.error("❌ Please enter a statement to analyze.")
            else:
                with st.spinner(f"🔬 Analyzing statement with {selected_model}..."):
                    # Enhanced analysis process
                    analyzer = NLPAnalyzer()
                    processed_text = analyzer.preprocess_text(user_statement)
                    
                    try:
                        # Vectorize text
                        X_text = st.session_state.trained_models.vectorizer.transform([processed_text])
                        
                        # Extract comprehensive features
                        linguistic_features = analyzer.extract_linguistic_features(user_statement)
                        sentiment_features = analyzer.analyze_sentiment(user_statement)
                        
                        # Prepare feature array
                        feature_array = [
                            linguistic_features['word_count'],
                            linguistic_features['sentence_count'], 
                            linguistic_features['char_count'],
                            linguistic_features['avg_word_length'],
                            linguistic_features['avg_sentence_length'],
                            linguistic_features['unique_word_ratio'],
                            linguistic_features['long_word_ratio'],
                            linguistic_features['lexical_diversity'],
                            sentiment_features['polarity'],
                            sentiment_features['subjectivity']
                        ]
                        
                        X_features = np.array([feature_array])
                        X_combined = np.hstack([X_text.toarray(), X_features])
                        
                        # Get prediction
                        model = st.session_state.trained_models.results[selected_model]['model']
                        prediction = model.predict(X_combined)[0]
                        probabilities = model.predict_proba(X_combined)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
                        confidence = probabilities[prediction]
                        
                        # Display enhanced results
                        st.markdown("---")
                        st.markdown("## 📋 Analysis Results")
                        
                        # Verdict card
                        if prediction == 1:
                            verdict = "✅ LIKELY TRUE"
                            verdict_color = "#28a745"
                            verdict_emoji = "✅"
                            credibility_score = confidence
                        else:
                            verdict = "❌ LIKELY FALSE"
                            verdict_color = "#dc3545" 
                            verdict_emoji = "❌"
                            credibility_score = 1 - confidence
                        
                        st.markdown(f"""
                        <div class='fact-check-card'>
                            <div style='text-align: center; margin-bottom: 2rem;'>
                                <h1 style='color: {verdict_color}; font-size: 3rem; margin: 0;'>{verdict_emoji}</h1>
                                <h2 style='color: {verdict_color}; margin: 0;'>{verdict}</h2>
                            </div>
                            
                            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;'>
                                <div style='text-align: center;'>
                                    <h3 style='color: #667eea; margin: 0;'>{credibility_score:.2%}</h3>
                                    <p style='margin: 0;'>Confidence Score</p>
                                </div>
                                <div style='text-align: center;'>
                                    <h3 style='color: #667eea; margin: 0;'>{selected_model}</h3>
                                    <p style='margin: 0;'>Analysis Model</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Humorous critique
                        humorous_verdict = generate_humorous_critique(
                            user_statement, prediction, confidence, 
                            {'word_count': linguistic_features['word_count'], 'sentiment': sentiment_features}
                        )
                        st.markdown(f'<div class="humor-section">{humorous_verdict}</div>', unsafe_allow_html=True)
                        
                        # Enhanced credibility assessment
                        st.markdown("### 🔍 Credibility Assessment")
                        
                        if credibility_score > 0.85:
                            assessment = "**🟢 High Credibility** - This statement appears highly reliable based on comprehensive analysis."
                        elif credibility_score > 0.70:
                            assessment = "**🟡 Moderate Credibility** - This statement seems plausible but verification with additional sources is recommended."
                        elif credibility_score > 0.55:
                            assessment = "**🟠 Low Credibility** - Exercise caution and verify this statement with trusted sources."
                        else:
                            assessment = "**🔴 Very Low Credibility** - This statement shows strong indicators of being unreliable or misleading."
                        
                        st.info(assessment)
                        
                        # Credibility progress with enhanced styling
                        st.markdown(f"**Overall Credibility Score:** {credibility_score:.2%}")
                        
                        # Enhanced progress bar with color coding
                        progress_color = (
                            "#28a745" if credibility_score > 0.85 else
                            "#ffc107" if credibility_score > 0.70 else  
                            "#fd7e14" if credibility_score > 0.55 else
                            "#dc3545"
                        )
                        
                        st.markdown(f"""
                        <div style='background: #e9ecef; border-radius: 10px; padding: 3px; margin: 1rem 0;'>
                            <div style='background: {progress_color}; width: {credibility_score * 100}%; 
                                     height: 20px; border-radius: 8px; transition: width 0.5s ease;'></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.info("💡 Please try training the models again or use a different statement.")

def show_about():
    """Enhanced About Section"""
    st.markdown('<div class="sub-header">ℹ️ About TruthDetector AI</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem; border-radius: 15px; color: white; margin-bottom: 2rem;'>
        <h2 style='color: white; margin: 0;'>🔍 TruthDetector AI</h2>
        <p style='color: white; font-size: 1.2rem; opacity: 0.9;'>
        Advanced Fact-Checking Platform using Natural Language Processing and Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology stack
    st.markdown("## 🛠️ Technology Stack")
    
    tech_cols = st.columns(4)
    
    with tech_cols[0]:
        st.markdown("""
        <div class='model-card' style='border-top-color: #667eea;'>
            <h4>🔤 Natural Language Processing</h4>
            <ul>
            <li>NLTK</li>
            <li>TextBlob</li>
            <li>TF-IDF Vectorization</li>
            <li>Sentiment Analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_cols[1]:
        st.markdown("""
        <div class='model-card' style='border-top-color: #4ecdc4;'>
            <h4>🤖 Machine Learning</h4>
            <ul>
            <li>Scikit-learn</li>
            <li>Multiple Classifiers</li>
            <li>Cross-validation</li>
            <li>Performance Metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_cols[2]:
        st.markdown("""
        <div class='model-card' style='border-top-color: #ff6b6b;'>
            <h4>📊 Data Visualization</h4>
            <ul>
            <li>Plotly</li>
            <li>Streamlit</li>
            <li>Interactive Charts</li>
            <li>Real-time Updates</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_cols[3]:
        st.markdown("""
        <div class='model-card' style='border-top-color: #ffd93d;'>
            <h4>🎯 Models Used</h4>
            <ul>
            <li>Decision Tree</li>
            <li>Logistic Regression</li>
            <li>Naive Bayes</li>
            <li>Support Vector Machine</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Important notes and disclaimer
    st.markdown("---")
    st.markdown("## ⚠️ Important Notes & Disclaimer")
    
    st.markdown("""
    <div class='warning-box'>
    <h4>🔒 Educational Purpose</h4>
    <p>This application is designed for <strong>educational and demonstration purposes only</strong>. 
    While it uses advanced machine learning techniques, it should not be relied upon for critical 
    fact-checking without additional verification.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>🧭 Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        app_section = st.radio(
            "Choose a section:",
            ["🏠 Dashboard", "📊 Data Collection", "🔤 NLP Analysis", 
             "📈 Model Performance", "🔍 Fact Checker", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Stats
        if st.session_state.scraped_data:
            st.markdown("### 📈 Quick Stats")
            true_count = len([s for s in st.session_state.scraped_data 
                            if s['rating'] in ['true', 'mostly-true']])
            false_count = len([s for s in st.session_state.scraped_data 
                             if s['rating'] in ['false', 'pants-fire', 'mostly-false']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("True", true_count)
            with col2:
                st.metric("False", false_count)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        <div class='info-box'>
        This is a demonstration app for educational purposes. 
        Always verify facts through multiple reliable sources.
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Sections
    if app_section == "🏠 Dashboard":
        show_dashboard()
    elif app_section == "📊 Data Collection":
        show_data_collection()
    elif app_section == "🔤 NLP Analysis":
        show_nlp_analysis()
    elif app_section == "📈 Model Performance":
        show_model_performance()
    elif app_section == "🔍 Fact Checker":
        show_fact_checker()
    else:  # About
        show_about()

if __name__ == "__main__":
    main()
