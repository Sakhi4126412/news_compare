import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Set page configuration
st.set_page_config(
    page_title="TruthDetector AI",
    page_icon="🔍",
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
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .humor-section {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        font-style: italic;
    }
    .phase-analysis {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class PolitifactScraper:
    def __init__(self):
        self.base_url = "https://www.politifact.com"
    
    def scrape_fact_checks(self, start_date, end_date, max_pages=5):
        """
        Scrape fact checks from PolitiFact within date range
        Note: This is a simplified version. Actual implementation may need to handle pagination and rate limiting.
        """
        st.info("🔍 Scraping PolitiFact data... This may take a few moments.")
        
        # Sample data for demonstration (in real implementation, replace with actual scraping)
        sample_data = [
            {
                'statement': 'The economy has created 15 million new jobs since I took office.',
                'speaker': 'Political Figure A',
                'date': '2024-01-15',
                'truth_value': 'false',
                'analysis': 'Exaggerated job numbers without context'
            },
            {
                'statement': 'Climate change is not caused by human activity.',
                'speaker': 'Political Figure B',
                'date': '2024-01-10',
                'truth_value': 'pants-fire',
                'analysis': 'Contradicts scientific consensus'
            },
            {
                'statement': 'The new policy will reduce healthcare costs for middle-class families.',
                'speaker': 'Political Figure C',
                'date': '2024-01-08',
                'truth_value': 'true',
                'analysis': 'Supported by independent analysis'
            },
            {
                'statement': 'Our border is completely secure and under control.',
                'speaker': 'Political Figure D',
                'date': '2024-01-05',
                'truth_value': 'false',
                'analysis': 'Contradicted by official statistics'
            },
            {
                'statement': 'The infrastructure bill will create 2 million new jobs.',
                'speaker': 'Political Figure E',
                'date': '2024-01-03',
                'truth_value': 'mostly-true',
                'analysis': 'Reasonable estimate based on economic models'
            }
        ]
        
        df = pd.DataFrame(sample_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by date range
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        filtered_df = df[mask]
        
        return filtered_df

class NLPAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def lexical_analysis(self, text):
        """Perform lexical analysis"""
        tokens = word_tokenize(text.lower())
        unique_tokens = set(tokens)
        
        return {
            'token_count': len(tokens),
            'unique_tokens': len(unique_tokens),
            'lexical_diversity': len(unique_tokens) / len(tokens) if tokens else 0,
            'avg_word_length': np.mean([len(token) for token in tokens]) if tokens else 0,
            'readability_score': textstat.flesch_reading_ease(text)
        }
    
    def syntactic_analysis(self, text):
        """Perform syntactic analysis"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        
        # Count parts of speech
        pos_counts = {}
        for word, pos in pos_tags:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        return {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'pos_distribution': pos_counts
        }
    
    def semantic_analysis(self, text):
        """Perform semantic analysis"""
        sentiment = self.sia.polarity_scores(text)
        
        # Simple semantic features
        modal_verbs = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
        words = word_tokenize(text.lower())
        modal_count = sum(1 for word in words if word in modal_verbs)
        
        return {
            'sentiment_compound': sentiment['compound'],
            'sentiment_positive': sentiment['pos'],
            'sentiment_negative': sentiment['neg'],
            'sentiment_neutral': sentiment['neu'],
            'modal_verb_count': modal_count
        }
    
    def discourse_analysis(self, text):
        """Perform discourse analysis"""
        sentences = sent_tokenize(text)
        
        # Simple discourse features
        discourse_markers = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 'nevertheless']
        words = [word.lower() for word in word_tokenize(text)]
        discourse_count = sum(1 for word in words if word in discourse_markers)
        
        return {
            'discourse_markers': discourse_count,
            'cohesion_score': min(discourse_count / len(sentences), 1) if sentences else 0
        }
    
    def pragmatic_analysis(self, text):
        """Perform pragmatic analysis"""
        # Analyze formality and contextual features
        formal_words = ['therefore', 'however', 'moreover', 'furthermore', 'consequently']
        informal_words = ['like', 'you know', 'actually', 'basically', 'literally']
        
        words = [word.lower() for word in word_tokenize(text)]
        formal_count = sum(1 for word in words if word in formal_words)
        informal_count = sum(1 for word in words if word in informal_words)
        
        return {
            'formality_score': formal_count / (formal_count + informal_count + 1),
            'contextual_complexity': len(text) / 100  # Simple proxy
        }
    
    def comprehensive_analysis(self, text):
        """Perform all NLP analyses"""
        return {
            'lexical': self.lexical_analysis(text),
            'syntactic': self.syntactic_analysis(text),
            'semantic': self.semantic_analysis(text),
            'discourse': self.discourse_analysis(text),
            'pragmatic': self.pragmatic_analysis(text)
        }

class TruthDetectionModel:
    def __init__(self):
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(random_state=42)
        }
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def prepare_features(self, df, nlp_phase):
        """Prepare features based on selected NLP phase"""
        features = []
        
        for text in df['statement']:
            analysis = NLPAnalyzer().comprehensive_analysis(text)
            phase_features = []
            
            if nlp_phase == 'lexical':
                phase_features = list(analysis['lexical'].values())
            elif nlp_phase == 'syntactic':
                phase_features = list(analysis['syntactic'].values())[:3]  # Exclude POS distribution
            elif nlp_phase == 'semantic':
                phase_features = list(analysis['semantic'].values())
            elif nlp_phase == 'discourse':
                phase_features = list(analysis['discourse'].values())
            elif nlp_phase == 'pragmatic':
                phase_features = list(analysis['pragmatic'].values())
            else:  # All phases
                all_features = []
                for phase in ['lexical', 'syntactic', 'semantic', 'discourse', 'pragmatic']:
                    phase_data = analysis[phase]
                    if isinstance(phase_data, dict):
                        all_features.extend([v for v in phase_data.values() if isinstance(v, (int, float))])
                phase_features = all_features
            
            features.append(phase_features)
        
        return np.array(features)
    
    def train_models(self, df, nlp_phase):
        """Train all models and return performance metrics"""
        # Prepare target variable (simplified truth values)
        truth_mapping = {'true': 1, 'mostly-true': 1, 'half-true': 0.5, 'false': 0, 'pants-fire': 0}
        y = df['truth_value'].map(truth_mapping).fillna(0)
        y_binary = (y > 0.5).astype(int)  # Binary classification
        
        X = self.prepare_features(df, nlp_phase)
        
        # Handle cases with no features
        if X.size == 0:
            st.warning("No features generated for the selected NLP phase. Please try a different phase.")
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'accuracy': accuracy,
                    'model': model,
                    'predictions': y_pred,
                    'true_labels': y_test
                }
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
                results[name] = {'accuracy': 0, 'model': None, 'predictions': [], 'true_labels': []}
        
        return results

def generate_humorous_critique(statement, truth_value, analysis):
    """Generate humorous critique of fact checks"""
    humor_templates = {
        'true': [
            "Well, well, well... someone actually told the truth! Alert the media! 🎉",
            "This statement is so true, it probably pays its taxes on time. Respect. 💯",
            "Truth detected! This claim has more credibility than my excuse for being late to work. 🕐"
        ],
        'mostly-true': [
            "Mostly true? So it's like a pizza with one pineapple slice - mostly good but with a questionable choice. 🍍",
            "This statement is telling the truth... with creative liberties! Like a biopic of a famous person. 🎬",
            "Mostly true - the factual equivalent of 'I'll be there in 5 minutes' when you're actually 7 minutes away. ⏰"
        ],
        'false': [
            "This claim is falser than a $3 bill in a monopoly game. 🎲",
            "If this statement were any less true, it would come with its own laugh track. 😂",
            "False alert! This claim has less truth than my gym membership usage statistics. 🏋️"
        ],
        'pants-fire': [
            "Pants on fire! This statement is so false, it could power a small city with its thermal energy. 🔥",
            "Warning: This claim may cause spontaneous combustion of trousers. Handle with extreme skepticism. 👖",
            "This isn't just false, it's 'call-the-fire-department' level of untruthfulness! 🚒"
        ]
    }
    
    templates = humor_templates.get(truth_value, humor_templates['false'])
    return np.random.choice(templates)

def create_visualizations(results, nlp_phase):
    """Create performance visualization for models"""
    if not results:
        st.warning("No results to visualize.")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Model accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax1.bar(model_names, accuracies, color=colors, alpha=0.8)
    ax1.set_title(f'Model Accuracy - {nlp_phase.upper()} Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Confusion matrix for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_result = results[best_model_name]
    
    if len(best_result['true_labels']) > 0 and len(best_result['predictions']) > 0:
        cm = confusion_matrix(best_result['true_labels'], best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
    
    # Feature importance visualization (placeholder)
    phases = ['Lexical', 'Syntactic', 'Semantic', 'Discourse', 'Pragmatic']
    phase_impact = [0.25, 0.20, 0.30, 0.15, 0.10]  # Example impacts
    
    ax3.pie(phase_impact, labels=phases, autopct='%1.1f%%', startangle=90, colors=colors)
    ax3.set_title('NLP Phase Impact Distribution', fontweight='bold')
    
    # Performance trend
    ax4.plot(model_names, accuracies, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    ax4.set_title('Model Performance Comparison', fontweight='bold')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 TruthDetector AI</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced NLP-powered Fact Verification System")
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.sidebar.title("Configuration")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # NLP phase selection
    nlp_phase = st.sidebar.selectbox(
        "Select NLP Analysis Phase",
        ["lexical", "syntactic", "semantic", "discourse", "pragmatic", "all"]
    )
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select ML Models",
        ["Decision Tree", "Logistic Regression", "Naive Bayes", "SVM"],
        default=["Decision Tree", "Logistic Regression", "Naive Bayes", "SVM"]
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🤖 Model Performance", "🎭 Fact Check", "📈 Insights"])
    
    with tab1:
        st.markdown('<div class="sub-header">Data Collection & NLP Analysis</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Scrape & Analyze Data"):
            with st.spinner("Processing data..."):
                # Scrape data
                scraper = PolitifactScraper()
                df = scraper.scrape_fact_checks(start_date, end_date)
                
                if df.empty:
                    st.warning("No data found for the selected date range.")
                    return
                
                # Perform NLP analysis
                analyzer = NLPAnalyzer()
                df['nlp_analysis'] = df['statement'].apply(analyzer.comprehensive_analysis)
                
                # Display data
                st.success(f"✅ Successfully analyzed {len(df)} fact checks!")
                
                # Show sample data
                st.subheader("Sample Fact Checks")
                display_df = df[['statement', 'speaker', 'date', 'truth_value']].copy()
                st.dataframe(display_df, use_container_width=True)
                
                # NLP Analysis Results
                st.subheader("NLP Phase Analysis")
                
                # Show analysis for first statement as example
                if len(df) > 0:
                    sample_analysis = df.iloc[0]['nlp_analysis']
                    
                    cols = st.columns(5)
                    phases = ['lexical', 'syntactic', 'semantic', 'discourse', 'pragmatic']
                    phase_names = ['Lexical', 'Syntactic', 'Semantic', 'Discourse', 'Pragmatic']
                    
                    for i, (col, phase, phase_name) in enumerate(zip(cols, phases, phase_names)):
                        with col:
                            st.markdown(f'<div class="phase-analysis"><strong>{phase_name}</strong>', unsafe_allow_html=True)
                            phase_data = sample_analysis[phase]
                            for key, value in list(phase_data.items())[:3]:
                                if isinstance(value, (int, float)):
                                    st.metric(key.replace('_', ' ').title(), f"{value:.3f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                
                # Store data in session state
                st.session_state.df = df
                st.session_state.analyzer = analyzer
    
    with tab2:
        st.markdown('<div class="sub-header">Model Performance Benchmark</div>', unsafe_allow_html=True)
        
        if 'df' not in st.session_state:
            st.info("Please scrape and analyze data first in the 'Data Analysis' tab.")
        else:
            df = st.session_state.df
            
            if st.button("🏋️ Train Models & Evaluate"):
                with st.spinner("Training models..."):
                    # Train models
                    model = TruthDetectionModel()
                    results = model.train_models(df, nlp_phase)
                    
                    if results:
                        # Display results
                        st.subheader("Performance Metrics")
                        
                        # Create metrics columns
                        cols = st.columns(len(results))
                        for i, (model_name, result) in enumerate(results.items()):
                            with cols[i]:
                                accuracy = result['accuracy']
                                st.metric(
                                    label=model_name,
                                    value=f"{accuracy:.3f}",
                                    delta="High" if accuracy > 0.7 else "Medium" if accuracy > 0.5 else "Low"
                                )
                        
                        # Create visualizations
                        st.subheader("Performance Visualization")
                        create_visualizations(results, nlp_phase)
                        
                        # Store results
                        st.session_state.results = results
                        st.session_state.nlp_phase = nlp_phase
    
    with tab3:
        st.markdown('<div class="sub-header">Fact Credibility Check</div>', unsafe_allow_html=True)
        
        # User input for fact checking
        st.markdown("### 🔎 Check Statement Credibility")
        user_statement = st.text_area(
            "Enter a statement to check:",
            placeholder="e.g., 'The moon is made of cheese'",
            height=100
        )
        
        if st.button("🔍 Analyze Credibility") and user_statement:
            with st.spinner("Analyzing statement..."):
                analyzer = NLPAnalyzer()
                
                # Perform comprehensive analysis
                analysis = analyzer.comprehensive_analysis(user_statement)
                
                # Display analysis results
                st.subheader("NLP Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Lexical Features**")
                    lexical = analysis['lexical']
                    st.write(f"• Token Count: {lexical['token_count']}")
                    st.write(f"• Unique Tokens: {lexical['unique_tokens']}")
                    st.write(f"• Readability Score: {lexical['readability_score']:.1f}")
                
                with col2:
                    st.markdown("**Semantic Features**")
                    semantic = analysis['semantic']
                    st.write(f"• Sentiment: {semantic['sentiment_compound']:.3f}")
                    st.write(f"• Modal Verbs: {semantic['modal_verb_count']}")
                
                # Generate credibility score (simplified)
                credibility_score = min(
                    (lexical['lexical_diversity'] * 0.3 +
                     abs(semantic['sentiment_compound']) * 0.3 +
                     analysis['pragmatic']['formality_score'] * 0.4) * 100, 100
                )
                
                # Display credibility score
                st.subheader("Credibility Assessment")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Credibility Score", f"{credibility_score:.1f}%")
                    
                    if credibility_score > 70:
                        st.success("✅ High Credibility")
                    elif credibility_score > 40:
                        st.warning("⚠️ Medium Credibility")
                    else:
                        st.error("❌ Low Credibility")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Humorous critique
                st.subheader("🎭 Humorous Analysis")
                truth_level = "true" if credibility_score > 70 else "mostly-true" if credibility_score > 40 else "false"
                critique = generate_humorous_critique(user_statement, truth_level, analysis)
                st.markdown(f'<div class="humor-section">{critique}</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="sub-header">Insights & Recommendations</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 📊 NLP Phase Insights
        
        **Lexical Analysis**: Examines vocabulary richness, word complexity, and readability
        - *Key Metrics*: Token count, lexical diversity, readability scores
        - *Impact*: Higher diversity often correlates with more nuanced statements
        
        **Syntactic Analysis**: Studies sentence structure and grammar
        - *Key Metrics*: Sentence length, part-of-speech distribution
        - *Impact*: Complex structures may indicate more carefully constructed claims
        
        **Semantic Analysis**: Analyzes meaning and sentiment
        - *Key Metrics*: Sentiment scores, modal verb usage
        - *Impact*: Extreme sentiments may correlate with exaggerated claims
        
        **Discourse Analysis**: Examines text cohesion and flow
        - *Key Metrics*: Discourse markers, coherence scores
        - *Impact*: Better cohesion often indicates more logical arguments
        
        **Pragmatic Analysis**: Considers context and intent
        - *Key Metrics*: Formality, contextual complexity
        - *Impact*: Formal language may indicate more serious claims
        """)
        
        st.markdown("""
        ### 🎯 User Prompt for Fact Checking
        
        When evaluating statements for credibility, consider:
        
        1. **Source Verification**
           - "What are the credentials of the speaker/organization?"
           - "Is there potential bias or conflict of interest?"
        
        2. **Evidence Quality**
           - "Are there specific data points or references?"
           - "Is the evidence recent and from reliable sources?"
        
        3. **Logical Consistency**
           - "Does the statement follow logical reasoning?"
           - "Are there any obvious contradictions?"
        
        4. **Context Analysis**
           - "What is the broader context of this statement?"
           - "Are important details being omitted?"
        
        5. **Corroboration**
           - "Do independent sources confirm this information?"
           - "Is there consensus among experts?"
        """)

if __name__ == "__main__":
    main()
