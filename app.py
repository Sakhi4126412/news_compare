import streamlit as st
import requests
import pandas as pd
import os

# ------------------------------------------------------
# ✅ Load API key safely (works both locally & in Streamlit Cloud)
# ------------------------------------------------------
API_KEY = st.secrets.get("GOOGLE_FACTCHECK_API_KEY", os.getenv("GOOGLE_FACTCHECK_API_KEY"))

if not API_KEY:
    st.error("🚨 Google Fact Check API key not found. Please add it in `.streamlit/secrets.toml` or as an environment variable.")
    st.stop()

# ------------------------------------------------------
# App Title and Description
# ------------------------------------------------------
st.title("📰 Fact Check Comparison App")
st.write("""
Compare the truthfulness of news from two different fact-checker APIs.
Enter a statement or headline, and we'll check its credibility using Google's Fact Check API and another API.
""")

# ------------------------------------------------------
# User Input
# ------------------------------------------------------
statement = st.text_input("Enter a news statement or claim:")

# ------------------------------------------------------
# Function to fetch from Google Fact Check API
# ------------------------------------------------------
def fetch_google_factcheck(claim):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={claim}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Error fetching from Google API: {response.status_code}"

    data = response.json().get("claims", [])
    results = []
    for claim in data:
        results.append({
            "Claim": claim.get("text", "N/A"),
            "Rating": claim.get("claimReview", [{}])[0].get("textualRating", "N/A"),
            "Publisher": claim.get("claimReview", [{}])[0].get("publisher", {}).get("name", "N/A"),
            "URL": claim.get("claimReview", [{}])[0].get("url", "N/A")
        })
    return pd.DataFrame(results) if results else None

# ------------------------------------------------------
# Placeholder: Second API (You can integrate another later)
# ------------------------------------------------------
def fetch_other_factcheck(claim):
    # Example dummy data
    return pd.DataFrame([
        {"Claim": claim, "Rating": "Likely True", "Publisher": "OtherFactCheckAPI", "URL": "https://example.com"}
    ])

# ------------------------------------------------------
# Run Fact Check
# ------------------------------------------------------
if st.button("🔍 Check Truthfulness") and statement:
    st.subheader("Google Fact Check Results")
    google_df = fetch_google_factcheck(statement)
    if google_df is not None:
        st.dataframe(google_df)
    else:
        st.info("No results found in Google Fact Check API.")

    st.subheader("Other Fact Check Results")
    other_df = fetch_other_factcheck(statement)
    st.dataframe(other_df)
