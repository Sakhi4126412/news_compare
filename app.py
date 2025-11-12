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
            
            # Row 1: Comparison and Radar
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_comparison, use_container_width=True)
            with col2:
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Row 2: Data distribution and confusion matrix
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_sunburst, use_container_width=True)
            with col2:
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # Row 3: Detailed metrics
            st.markdown("#### 📋 Detailed Performance Metrics")
            
            # Create performance table
            performance_data = []
            for model_name, result in st.session_state.results.items():
                # Calculate additional metrics
                cm = result['confusion_matrix']
                tn, fp, fn, tp = cm.ravel()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                performance_data.append({
                    'Model': model_name,
                    'Accuracy': f"{result['accuracy']:.3f}",
                    'Precision': f"{precision:.3f}",
                    'Recall': f"{recall:.3f}",
                    'F1-Score': f"{f1:.3f}",
                    'Training Time': f"{result['training_time']:.2f}s",
                    'Status': '✅ Trained'
                })
            
            # Display as styled dataframe
            df_performance = pd.DataFrame(performance_data)
            st.dataframe(
                df_performance.style.background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                                                       cmap='Blues'),
                use_container_width=True
            )
            
            # Best model highlight
            best_model_name = max(st.session_state.results.items(), key=lambda x: x[1]['accuracy'])[0]
            best_accuracy = st.session_state.results[best_model_name]['accuracy']
            
            st.markdown(f"""
            <div class='success-box'>
                <h3>🎯 Best Performing Model: {best_model_name}</h3>
                <p>With an accuracy of <strong>{best_accuracy:.3f}</strong>, this model provides the most reliable predictions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model insights
            st.markdown("#### 💡 Model Insights")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class='feature-card'>
                <h4>🏆 Decision Tree</h4>
                <p><strong>Strengths:</strong> Easy to interpret, handles non-linear relationships</p>
                <p><strong>Best for:</strong> Clear rule-based patterns</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='feature-card'>
                <h4>📊 Logistic Regression</h4>
                <p><strong>Strengths:</strong> Probabilistic outputs, fast training</p>
                <p><strong>Best for:</strong> Linear relationships, confidence scores</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='feature-card'>
                <h4>🎯 Naive Bayes</h4>
                <p><strong>Strengths:</strong> Fast, works well with text data</p>
                <p><strong>Best for:</strong> Text classification, quick predictions</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='feature-card'>
                <h4>🚀 SVM</h4>
                <p><strong>Strengths:</strong> Handles high-dimensional data well</p>
                <p><strong>Best for:</strong> Complex patterns, clear margin separation</p>
                </div>
                """, unsafe_allow_html=True)

def show_fact_checker():
    """Enhanced Fact Checker Section"""
    st.markdown('<div class="sub-header">🔍 Fact Checking Tool</div>', unsafe_allow_html=True)
    
    if st.session_state.trained_models is None:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("Please train the models first in the 'Model Performance' section.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 Train Models Now"):
                with st.spinner("Training models with sample data..."):
                    detector = TruthDetector()
                    X, y, _ = detector.prepare_data(st.session_state.scraped_data)
                    detector.train_models(X, y)
                    st.session_state.trained_models = detector
                    st.session_state.results = detector.results
                st.rerun()
        
        with col2:
            if st.button("📥 Load Sample Data First"):
                scraper = PolitifactScraper()
                st.session_state.scraped_data = scraper.get_comprehensive_sample_data()
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
            
            analysis_depth = st.selectbox(
                "Analysis Depth:",
                ["Standard", "Detailed", "Comprehensive"],
                index=0
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
                            recommendations = [
                                "✅ Can be considered trustworthy",
                                "✅ Supported by linguistic patterns", 
                                "✅ Consistent with factual statements"
                            ]
                        elif credibility_score > 0.70:
                            assessment = "**🟡 Moderate Credibility** - This statement seems plausible but verification with additional sources is recommended."
                            recommendations = [
                                "⚠️ Verify with reputable sources",
                                "⚠️ Check for supporting evidence",
                                "✅ Shows some reliable patterns"
                            ]
                        elif credibility_score > 0.55:
                            assessment = "**🟠 Low Credibility** - Exercise caution and verify this statement with trusted sources."
                            recommendations = [
                                "❌ Requires external verification",
                                "⚠️ Limited reliability indicators",
                                "🔍 Investigate further before sharing"
                            ]
                        else:
                            assessment = "**🔴 Very Low Credibility** - This statement shows strong indicators of being unreliable or misleading."
                            recommendations = [
                                "❌ High risk of misinformation",
                                "❌ Multiple unreliable indicators",
                                "🚫 Avoid sharing without verification"
                            ]
                        
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
                        
                        # Recommendations
                        st.markdown("#### 📋 Recommendations")
                        for rec in recommendations:
                            st.write(f"- {rec}")
                        
                        # Detailed analysis section
                        st.markdown("---")
                        st.markdown("### 🔬 Detailed Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### 📝 Linguistic Analysis")
                            
                            # Create feature gauges
                            features_to_display = {
                                'Word Count': (linguistic_features['word_count'], 0, 100),
                                'Sentence Count': (linguistic_features['sentence_count'], 0, 10),
                                'Avg Word Length': (linguistic_features['avg_word_length'], 0, 10),
                                'Lexical Diversity': (linguistic_features['lexical_diversity'], 0, 1)
                            }
                            
                            for feature_name, (value, min_val, max_val) in features_to_display.items():
                                normalized_value = (value - min_val) / (max_val - min_val)
                                st.metric(feature_name, f"{value:.1f}")
                                st.progress(min(normalized_value, 1.0))
                        
                        with col2:
                            st.markdown("##### 😊 Sentiment & Style")
                            
                            # Sentiment indicators
                            sentiment_value = sentiment_features['polarity']
                            subjectivity_value = sentiment_features['subjectivity']
                            
                            st.metric("Sentiment Polarity", f"{sentiment_value:.2f}")
                            st.metric("Subjectivity", f"{subjectivity_value:.2f}")
                            st.metric("Sentiment Label", sentiment_features['label'])
                            
                            # Style assessment
                            if linguistic_features['word_count'] < 10:
                                style_assessment = "📝 Concise"
                            elif linguistic_features['word_count'] > 50:
                                style_assessment = "📚 Detailed"  
                            else:
                                style_assessment = "📄 Balanced"
                                
                            st.metric("Writing Style", style_assessment)
                        
                        # User guidance section
                        st.markdown("---")
                        st.markdown("### 💡 Fact-Checking Guidance")
                        
                        guidance_cols = st.columns(2)
                        
                        with guidance_cols[0]:
                            st.markdown("""
                            <div class='feature-card'>
                            <h4>🔍 Source Verification</h4>
                            <ul>
                            <li>Check the original source of information</li>
                            <li>Verify author credentials and expertise</li>
                            <li>Look for supporting evidence and citations</li>
                            <li>Check publication date and context</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class='feature-card'>
                            <h4>🎯 Logical Analysis</h4>
                            <ul>
                            <li>Does the statement make logical sense?</li>
                            <li>Are there internal contradictions?</li>
                            <li>Does it align with established facts?</li>
                            <li>Consider the context and timing</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with guidance_cols[1]:
                            st.markdown("""
                            <div class='feature-card'>
                            <h4>🌐 External Corroboration</h4>
                            <ul>
                            <li>Check multiple independent sources</li>
                            <li>Look for expert consensus</li>
                            <li>Verify with fact-checking organizations</li>
                            <li>Consult academic or official sources</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class='feature-card'>
                            <h4>⚠️ Critical Thinking</h4>
                            <ul>
                            <li>Consider potential biases</li>
                            <li>Evaluate emotional language use</li>
                            <li>Check for oversimplification</li>
                            <li>Look for missing context</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.info("💡 Please try training the models again or use a different statement.")

def show_about():
    """Enhanced About Section"""
    st.markdown('<div class="sub-header">ℹ️ About TruthDetector AI</div>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 3rem; border-radius: 15px; color: white;'>
            <h2 style='color: white; margin: 0;'>🔍 TruthDetector AI</h2>
            <p style='color: white; font-size: 1.2rem; opacity: 0.9;'>
            Advanced Fact-Checking Platform using Natural Language Processing and Machine Learning
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3>🚀 Version 2.0</h3>
            <p><strong>Enhanced Visual Analytics</strong></p>
            <p><strong>Multi-Model Approach</strong></p>
            <p><strong>Real-time Analysis</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("## 🛠️ How It Works")
    
    steps = [
        {
            "icon": "📊",
            "title": "Data Collection",
            "description": "Gather fact-checked statements from reliable sources and prepare them for analysis."
        },
        {
            "icon": "🔤", 
            "title": "NLP Processing",
            "description": "Analyze text features, sentiment, linguistic patterns, and complexity metrics."
        },
        {
            "icon": "🤖",
            "title": "Model Training", 
            "description": "Train multiple machine learning models to recognize patterns of truth and falsehood."
        },
        {
            "icon": "🔍",
            "title": "Fact Checking",
            "description": "Apply trained models to new statements and provide credibility assessments."
        }
    ]
    
    cols = st.columns(4)
    for idx, step in enumerate(steps):
        with cols[idx]:
            st.markdown(f"""
            <div class='feature-card'>
                <div style='font-size: 2.5rem; text-align: center;'>{step['icon']}</div>
                <h4 style='text-align: center;'>{step['title']}</h4>
                <p style='text-align: center; font-size: 0.9rem;'>{step['description']}</p>
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
    
    # Model details
    st.markdown("## 🎯 Machine Learning Models")
    
    model_details = [
        {
            "name": "Decision Tree",
            "icon": "🌳",
            "description": "Rule-based classification that creates a tree-like model of decisions",
            "strengths": ["Easy to interpret", "Handles non-linear data", "No feature scaling needed"],
            "best_for": "Clear decision boundaries"
        },
        {
            "name": "Logistic Regression", 
            "icon": "📈",
            "description": "Statistical model that uses probability for classification",
            "strengths": ["Probabilistic outputs", "Fast training", "Good for linear relationships"],
            "best_for": "Confidence scoring"
        },
        {
            "name": "Naive Bayes",
            "icon": "🎯", 
            "description": "Probabilistic classifier based on Bayes' theorem with independence assumptions",
            "strengths": ["Very fast", "Works well with text", "Handles high dimensions"],
            "best_for": "Text classification"
        },
        {
            "name": "Support Vector Machine",
            "icon": "🚀",
            "description": "Finds optimal hyperplane to separate classes in high-dimensional space", 
            "strengths": ["Effective in high dimensions", "Memory efficient", "Versatile kernels"],
            "best_for": "Complex patterns"
        }
    ]
    
    for model in model_details:
        with st.expander(f"{model['icon']} {model['name']}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {model['description']}")
                st.markdown("**Strengths:**")
                for strength in model['strengths']:
                    st.markdown(f"- {strength}")
            
            with col2:
                st.markdown("**Best For:**")
                st.info(model['best_for'])
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <h4>📝 Limitations</h4>
        <ul>
        <li>Uses sample data for demonstration</li>
        <li>Accuracy depends on training data quality</li>
        <li>May not capture all contextual nuances</li>
        <li>Limited to English language analysis</li>
        <li>Cannot verify factual accuracy directly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
        <h4>✅ Best Practices</h4>
        <ul>
        <li>Always verify with multiple sources</li>
        <li>Check reputable fact-checking organizations</li>
        <li>Consider source credibility and expertise</li>
        <li>Look for supporting evidence</li>
        <li>Be aware of potential biases</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # User guidance for fact-checking
    st.markdown("## 💡 User Guidance for Fact-Checking")
    
    guidance_points = [
        ("🔍 Source Evaluation", "Check the credibility and expertise of information sources"),
        ("📚 Multiple Verification", "Always cross-reference with multiple independent sources"),
        ("🎯 Context Matters", "Consider the context, timing, and purpose of the information"),
        ("🤔 Critical Thinking", "Apply logical reasoning and watch for emotional manipulation"),
        ("📊 Evidence-Based", "Look for verifiable evidence and data supporting claims"),
        ("🌐 Expert Consensus", "Check what experts in the field generally agree on"),
        ("⚠️ Bias Awareness", "Be aware of potential biases in sources and information"),
        ("🚨 Red Flags", "Watch for sensationalism, lack of sources, or logical fallacies")
    ]
    
    for i in range(0, len(guidance_points), 2):
        col1, col2 = st.columns(2)
        with col1:
            point = guidance_points[i]
            st.markdown(f"**{point[0]}**\n\n{point[1]}")
        with col2:
            if i + 1 < len(guidance_points):
                point = guidance_points[i + 1]
                st.markdown(f"**{point[0]}**\n\n{point[1]}")
    
    # Final note
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;'>
        <h3>🔍 Stay Informed, Stay Critical</h3>
        <p>TruthDetector AI is a tool to assist in information evaluation, but critical thinking 
        and multiple source verification remain essential for accurate fact-checking.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
