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

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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
</style>
""", unsafe_allow_html=True)

class PolitifactScraper:
    def __init__(self):
        self.base_url = "https://www.politifact.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_facts(self, start_date, end_date, max_pages=5):
        """Scrape Politifact data within date range"""
        facts_data = []
        
        try:
            for page in range(1, max_pages + 1):
                url = f"{self.base_url}/factchecks/list/?page={page}"
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                fact_checks = soup.find_all('div', class_='m-statement__body')
                
                for fact in fact_checks:
                    try:
                        # Extract statement
                        statement_elem = fact.find('div', class_='m-statement__quote')
                        statement = statement_elem.text.strip() if statement_elem else ""
                        
                        # Extract rating
                        rating_elem = fact.find('div', class_='m-statement__meter')
                        if rating_elem:
                            rating_img = rating_elem.find('img')
                            if rating_img and 'alt' in rating_img.attrs:
                                rating = rating_img['alt']
                            else:
                                rating = "Unknown"
                        else:
                            rating = "Unknown"
                        
                        # Extract date
                        date_elem = fact.find('div', class_='m-statement__desc')
                        date_text = date_elem.text.strip() if date_elem else ""
                        
                        # Simple date parsing (you might need to enhance this)
                        date_match = re.search(r'(\w+ \d+, \d{4})', date_text)
                        if date_match:
                            fact_date = datetime.strptime(date_match.group(1), '%B %d, %Y')
                            
                            # Check if date is within range
                            if start_date <= fact_date <= end_date:
                                facts_data.append({
                                    'statement': statement,
                                    'rating': rating,
                                    'date': fact_date,
                                    'source': 'Politifact'
                                })
                        
                    except Exception as e:
                        continue
                
                # Add delay to be respectful to the server
                time.sleep(1)
                        
        except Exception as e:
            st.error(f"Error scraping data: {str(e)}")
        
        return facts_data

class NLPAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    def extract_features(self, text):
        """Extract basic text features"""
        words = text.split()
        return {
            'word_count': len(words),
            'char_count': len(text),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'sentiment': self.analyze_sentiment(text)
        }

class TruthDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.results = {}
    
    def prepare_data(self, data):
        """Prepare data for training"""
        df = data.copy()
        
        # Preprocess text
        analyzer = NLPAnalyzer()
        df['processed_text'] = df['statement'].apply(analyzer.preprocess_text)
        
        # Convert ratings to binary (True vs False)
        true_ratings = ['true', 'mostly-true', 'half-true']
        df['is_true'] = df['rating'].apply(lambda x: 1 if any(true in x.lower() for true in true_ratings) else 0)
        
        # Vectorize text
        X_text = self.vectorizer.fit_transform(df['processed_text'])
        
        # Add additional features
        features = []
        for text in df['statement']:
            feat = analyzer.extract_features(text)
            features.append([feat['word_count'], feat['char_count'], feat['avg_word_length'], feat['sentiment']])
        
        X_features = np.array(features)
        X_combined = np.hstack([X_text.toarray(), X_features])
        y = df['is_true']
        
        return X_combined, y, df
    
    def train_models(self, X, y):
        """Train all models and store results"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred
            }
        
        return self.results

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
    """Create performance visualizations"""
    
    # Model accuracy comparison
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
    
    # Rating distribution
    rating_counts = data['rating'].value_counts()
    fig2 = px.pie(values=rating_counts.values, 
                  names=rating_counts.index,
                  title='Fact Check Rating Distribution',
                  color_discrete_sequence=px.colors.sequential.Blues_r)
    
    # Confusion matrix for best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
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
    
    return fig1, fig2, fig3

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
    
    # Sample data for demonstration (in real app, this would be scraped)
    @st.cache_data
    def load_sample_data():
        sample_data = [
            {'statement': 'The economy has grown significantly in the past year.', 'rating': 'true', 'date': datetime(2024, 1, 15)},
            {'statement': 'Climate change is a hoax created by scientists.', 'rating': 'false', 'date': datetime(2024, 1, 10)},
            {'statement': 'Vaccines are completely safe and effective for everyone.', 'rating': 'mostly-true', 'date': datetime(2024, 1, 5)},
            {'statement': 'The moon landing was filmed in a Hollywood studio.', 'rating': 'false', 'date': datetime(2024, 1, 1)},
            {'statement': 'Regular exercise improves mental health.', 'rating': 'true', 'date': datetime(2023, 12, 28)},
            {'statement': 'Eating carrots improves night vision dramatically.', 'rating': 'half-true', 'date': datetime(2023, 12, 25)},
            {'statement': 'The Earth is flat and stationary.', 'rating': 'false', 'date': datetime(2023, 12, 20)},
            {'statement': 'Drinking 8 glasses of water daily is essential for health.', 'rating': 'mostly-true', 'date': datetime(2023, 12, 15)},
        ]
        return pd.DataFrame(sample_data)
    
    if app_section == "Data Collection":
        st.markdown('<div class="sub-header">📊 Data Collection from Politifact</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime(2023, 12, 1))
        with col2:
            end_date = st.date_input("End Date", datetime(2024, 1, 15))
        
        if st.button("Scrape Politifact Data"):
            with st.spinner("Scraping data from Politifact... This may take a while."):
                scraper = PolitifactScraper()
                scraped_data = scraper.scrape_facts(start_date, end_date, max_pages=3)
                
                if scraped_data:
                    st.session_state.scraped_data = pd.DataFrame(scraped_data)
                    st.success(f"Successfully scraped {len(scraped_data)} fact checks!")
                else:
                    st.warning("No data found for the selected date range. Using sample data for demonstration.")
                    st.session_state.scraped_data = load_sample_data()
        
        if st.session_state.scraped_data is not None:
            st.subheader("Scraped Data Preview")
            st.dataframe(st.session_state.scraped_data, use_container_width=True)
            
            # Data statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Statements", len(st.session_state.scraped_data))
            with col2:
                true_count = len(st.session_state.scraped_data[
                    st.session_state.scraped_data['rating'].str.contains('true', case=False)
                ])
                st.metric("True Statements", true_count)
            with col3:
                false_count = len(st.session_state.scraped_data) - true_count
                st.metric("False Statements", false_count)
    
    elif app_section == "NLP Analysis":
        st.markdown('<div class="sub-header">🔤 Natural Language Processing Analysis</div>', unsafe_allow_html=True)
        
        if st.session_state.scraped_data is None:
            st.info("Please scrape data first in the 'Data Collection' section. Using sample data for demonstration.")
            st.session_state.scraped_data = load_sample_data()
        
        analyzer = NLPAnalyzer()
        
        # Show NLP features
        st.subheader("Text Analysis Features")
        
        sample_text = st.selectbox("Select a statement to analyze:", 
                                  st.session_state.scraped_data['statement'].tolist())
        
        if sample_text:
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
                for feature, value in features.items():
                    st.write(f"- {feature.replace('_', ' ').title()}: {value:.2f}")
                
                sentiment = analyzer.analyze_sentiment(sample_text)
                sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                st.write(f"- Sentiment: {sentiment_label} ({sentiment:.2f})")
    
    elif app_section == "Model Performance":
        st.markdown('<div class="sub-header">📈 Machine Learning Model Performance</div>', unsafe_allow_html=True)
        
        if st.session_state.scraped_data is None:
            st.info("Please scrape data first in the 'Data Collection' section. Using sample data for demonstration.")
            st.session_state.scraped_data = load_sample_data()
        
        if st.button("Train Models"):
            with st.spinner("Training machine learning models..."):
                detector = TruthDetector()
                X, y, processed_data = detector.prepare_data(st.session_state.scraped_data)
                results = detector.train_models(X, y)
                
                st.session_state.trained_models = detector
                st.session_state.results = results
                
                st.success("Models trained successfully!")
        
        if st.session_state.results:
            # Display results
            st.subheader("Model Accuracy Scores")
            
            # Create visualizations
            fig1, fig2, fig3 = create_visualizations(st.session_state.results, st.session_state.scraped_data)
            
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
            
            st.table(pd.DataFrame(performance_data))
    
    elif app_section == "Fact Checker":
        st.markdown('<div class="sub-header">🔍 Fact Checking Tool</div>', unsafe_allow_html=True)
        
        if st.session_state.trained_models is None:
            st.warning("Please train the models first in the 'Model Performance' section.")
            st.session_state.scraped_data = load_sample_data()
            
            # Train with sample data for demonstration
            detector = TruthDetector()
            X, y, _ = detector.prepare_data(st.session_state.scraped_data)
            detector.train_models(X, y)
            st.session_state.trained_models = detector
            st.session_state.results = detector.results
        
        st.markdown("### Check Statement Credibility")
        
        # User input
        user_statement = st.text_area("Enter a statement to check:", 
                                     "The Earth revolves around the Sun.")
        
        selected_model = st.selectbox("Select NLP Model for Analysis:",
                                     list(st.session_state.trained_models.models.keys()))
        
        if st.button("Analyze Credibility"):
            with st.spinner("Analyzing statement..."):
                # Preprocess and predict
                analyzer = NLPAnalyzer()
                processed_text = analyzer.preprocess_text(user_statement)
                
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
                    st.write(f"**Confidence:** {probability:.2%}")
                    st.write(f"**Model Used:** {selected_model}")
                
                with col2:
                    features = analyzer.extract_features(user_statement)
                    st.write("**Analysis Details:**")
                    st.write(f"- Word Count: {features['word_count']}")
                    st.write(f"- Sentiment: {features['sentiment']:.2f}")
                    st.write(f"- Characters: {features['char_count']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Humorous critique
                humorous_verdict = generate_humorous_critique(user_statement, prediction, probability)
                st.markdown(f'<div class="humor-section">{humorous_verdict}</div>', unsafe_allow_html=True)
                
                # Credibility assessment
                st.subheader("🔍 Credibility Assessment")
                
                credibility_score = probability if prediction == 1 else 1 - probability
                
                if credibility_score > 0.8:
                    assessment = "**High Credibility** - This statement appears reliable based on our analysis."
                elif credibility_score > 0.6:
                    assessment = "**Moderate Credibility** - This statement seems plausible but verify with additional sources."
                else:
                    assessment = "**Low Credibility** - Exercise caution and verify this statement with trusted sources."
                
                st.info(assessment)
                
                # Progress bar for credibility score
                st.write(f"Credibility Score: {credibility_score:.2%}")
                st.progress(float(credibility_score))
    
    else:  # About section
        st.markdown('<div class="sub-header">ℹ️ About TruthDetector AI</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### How It Works
        
        **TruthDetector AI** uses advanced Natural Language Processing and Machine Learning to analyze statements and assess their credibility.
        
        ### Methodology
        
        1. **Data Collection**: Scrapes fact-checked statements from Politifact
        2. **NLP Processing**: Analyzes text features, sentiment, and linguistic patterns
        3. **Machine Learning**: Trains multiple models to detect truth patterns
        4. **Credibility Assessment**: Provides confidence scores and humorous insights
        
        ### Models Used
        
        - **Decision Tree**: Rule-based classification
        - **Logistic Regression**: Statistical probability modeling
        - **Naive Bayes**: Probabilistic classification
        - **SVM**: Advanced pattern recognition
        
        ### Important Note
        
        This tool is for educational and demonstration purposes. Always verify important information through multiple reliable sources.
        
        ### User Prompt for Fact Checking
        
        When using this tool, consider:
        - The source of the information
        - Corroborating evidence from multiple sources
        - The context and timing of the statement
        - Potential biases in the statement
        - Whether the statement makes logical sense
        """)

if __name__ == "__main__":
    main()
