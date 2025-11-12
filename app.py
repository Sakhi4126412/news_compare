def show_fact_checker():
    """Enhanced Fact Checker Section"""
    st.markdown('<div class="sub-header">🔍 Fact Checking Tool</div>', unsafe_allow_html=True)
    
    if st.session_state.scraped_data is None:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("Please load sample data first in the 'Data Collection' section.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("📥 Load Sample Data Now"):
            scraper = PolitifactScraper()
            st.session_state.scraped_data = scraper.get_comprehensive_sample_data()
            st.rerun()
        return
    
    if st.session_state.trained_models is None:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("Please train the models first in the 'Model Performance' section.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 Train Models Now", use_container_width=True):
                with st.spinner("Training models with sample data..."):
                    try:
                        detector = TruthDetector()
                        X, y, processed_data = detector.prepare_data(st.session_state.scraped_data)
                        if X is not None and y is not None:
                            results = detector.train_models(X, y)
                            st.session_state.trained_models = detector
                            st.session_state.results = results
                            st.success("✅ Models trained successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to prepare data for training")
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
        return

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
        
        # Show model info
        if selected_model in st.session_state.results:
            accuracy = st.session_state.results[selected_model]['accuracy']
            st.metric("Model Accuracy", f"{accuracy:.3f}")

    # Analyze button
    if st.button("🔍 Analyze Credibility", type="primary", use_container_width=True):
        if not user_statement.strip():
            st.error("❌ Please enter a statement to analyze.")
        else:
            with st.spinner(f"🔬 Analyzing statement with {selected_model}..."):
                try:
                    analyzer = NLPAnalyzer()
                    
                    # Step 1: Preprocess the text
                    processed_text = analyzer.preprocess_text(user_statement)
                    
                    # Step 2: Extract features
                    linguistic_features = analyzer.extract_linguistic_features(user_statement)
                    sentiment_features = analyzer.analyze_sentiment(user_statement)
                    
                    # Step 3: Vectorize the text using the trained vectorizer
                    X_text = st.session_state.trained_models.vectorizer.transform([processed_text])
                    
                    # Step 4: Create feature array matching training structure
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
                    
                    # Convert to numpy array and ensure correct shape
                    X_features = np.array(feature_array).reshape(1, -1)
                    
                    # Step 5: Combine text features with linguistic features
                    # Ensure dimensions match by using only the linguistic features if there's a dimension mismatch
                    try:
                        X_combined = np.hstack([X_text.toarray(), X_features])
                    except ValueError as e:
                        st.warning("⚠️ Feature dimension mismatch. Using linguistic features only.")
                        X_combined = X_features
                    
                    # Step 6: Get prediction from the selected model
                    model = st.session_state.trained_models.results[selected_model]['model']
                    
                    # Handle different model types
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_combined)[0]
                        prediction = model.predict(X_combined)[0]
                        confidence = probabilities[prediction]
                    else:
                        # For models without probability estimates
                        prediction = model.predict(X_combined)[0]
                        confidence = 0.75  # Default confidence for models without proba
                    
                    # Step 7: Display results
                    st.markdown("---")
                    st.markdown("## 📋 Analysis Results")
                    
                    # Verdict card
                    if prediction == 1:
                        verdict = "✅ LIKELY TRUE"
                        verdict_color = "#28a745"
                        verdict_emoji = "✅"
                        credibility_score = confidence
                        explanation = "The statement shows patterns similar to verified true statements in our training data."
                    else:
                        verdict = "❌ LIKELY FALSE"
                        verdict_color = "#dc3545" 
                        verdict_emoji = "❌"
                        credibility_score = 1 - confidence if hasattr(model, 'predict_proba') else 0.25
                        explanation = "The statement shows patterns similar to debunked false statements in our training data."
                    
                    st.markdown(f"""
                    <div class='fact-check-card'>
                        <div style='text-align: center; margin-bottom: 2rem;'>
                            <h1 style='color: {verdict_color}; font-size: 3rem; margin: 0;'>{verdict_emoji}</h1>
                            <h2 style='color: {verdict_color}; margin: 0;'>{verdict}</h2>
                        </div>
                        
                        <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;'>
                            <div style='text-align: center;'>
                                <h3 style='color: #667eea; margin: 0;'>{credibility_score:.2%}</h3>
                                <p style='margin: 0; font-size: 0.9rem;'>Confidence Score</p>
                            </div>
                            <div style='text-align: center;'>
                                <h3 style='color: #667eea; margin: 0;'>{selected_model}</h3>
                                <p style='margin: 0; font-size: 0.9rem;'>Analysis Model</p>
                            </div>
                            <div style='text-align: center;'>
                                <h3 style='color: #667eea; margin: 0;'>{linguistic_features['word_count']}</h3>
                                <p style='margin: 0; font-size: 0.9rem;'>Words Analyzed</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Explanation
                    st.info(f"**Analysis Insight:** {explanation}")
                    
                    # Humorous critique
                    humorous_verdict = generate_humorous_critique(
                        user_statement, prediction, credibility_score, 
                        {
                            'word_count': linguistic_features['word_count'], 
                            'sentiment': sentiment_features
                        }
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
                        
                        # Create feature display
                        features_to_display = {
                            'Word Count': linguistic_features['word_count'],
                            'Sentence Count': linguistic_features['sentence_count'],
                            'Characters': linguistic_features['char_count'],
                            'Avg Word Length': f"{linguistic_features['avg_word_length']:.1f}",
                            'Avg Sentence Length': f"{linguistic_features['avg_sentence_length']:.1f}",
                            'Unique Words': f"{linguistic_features['unique_word_ratio']:.1%}",
                            'Long Words': f"{linguistic_features['long_word_ratio']:.1%}",
                            'Lexical Diversity': f"{linguistic_features['lexical_diversity']:.2f}"
                        }
                        
                        for feature_name, value in features_to_display.items():
                            st.metric(feature_name, value)
                    
                    with col2:
                        st.markdown("##### 😊 Sentiment & Style")
                        
                        # Sentiment indicators
                        sentiment_value = sentiment_features['polarity']
                        subjectivity_value = sentiment_features['subjectivity']
                        
                        st.metric("Sentiment Polarity", f"{sentiment_value:.3f}")
                        st.metric("Subjectivity", f"{subjectivity_value:.3f}")
                        st.metric("Sentiment Label", sentiment_features['label'])
                        
                        # Style assessment
                        word_count = linguistic_features['word_count']
                        if word_count < 15:
                            style_assessment = "📝 Concise"
                            style_explanation = "Short and direct statement"
                        elif word_count > 50:
                            style_assessment = "📚 Detailed"
                            style_explanation = "Comprehensive and elaborate statement"
                        else:
                            style_assessment = "📄 Balanced"
                            style_explanation = "Moderate length statement"
                            
                        st.metric("Writing Style", style_assessment)
                        st.caption(style_explanation)
                        
                        # Complexity assessment
                        complexity_score = (
                            linguistic_features['lexical_diversity'] + 
                            linguistic_features['long_word_ratio'] +
                            linguistic_features['unique_word_ratio']
                        ) / 3
                        
                        if complexity_score > 0.6:
                            complexity = "🎓 Advanced"
                        elif complexity_score > 0.3:
                            complexity = "📖 Intermediate"
                        else:
                            complexity = "🔤 Basic"
                            
                        st.metric("Complexity Level", complexity)
                    
                    # Feature visualization
                    st.markdown("##### 📊 Feature Radar Chart")
                    fig_radar, _, _ = create_feature_analysis_visualization(analyzer, user_statement)
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
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
                    st.error(f"❌ Analysis Error: {str(e)}")
                    st.markdown("""
                    <div class='warning-box'>
                    <h4>🛠️ Troubleshooting Tips</h4>
                    <ul>
                    <li>Make sure models are properly trained in the Model Performance section</li>
                    <li>Try using a different statement</li>
                    <li>Ensure the statement is in English</li>
                    <li>Check that the statement has sufficient content (at least 5 words)</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Debug information
                    with st.expander("🔧 Technical Details (for debugging)"):
                        st.write(f"Error type: {type(e).__name__}")
                        st.write(f"Error message: {str(e)}")
                        if st.session_state.trained_models:
                            st.write("Vectorizer fitted:", hasattr(st.session_state.trained_models.vectorizer, 'vocabulary_'))
                            if st.session_state.results:
                                st.write("Available models:", list(st.session_state.results.keys()))
