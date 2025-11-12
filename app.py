import streamlit as st
import requests
import pandas as pd
import os
import urllib.parse

# ----------------------------------------
# App Configuration
# ----------------------------------------
st.set_page_config(page_title="News Truth Comparison", layout="wide")
st.title("📰 News Truthfulness Comparison App")
st.write("Compare fact-check results from multiple sources using the **Google Fact Check Tools API**.")

# ----------------------------------------
# Load API Key Securely
# ----------------------------------------
API_KEY = None
if "GOOGLE_FACTCHECK_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_FACTCHECK_API_KEY"]
elif os.getenv("GOOGLE_FACTCHECK_API_KEY"):
    API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

if not API_KEY:
    st.error("🚨 Google Fact Check API key not found. Please add it in `.streamlit/secrets.toml` or as an environment variable.")
    st.stop()

# ----------------------------------------
# Input Field
# ----------------------------------------
user_query = st.text_input("🔍 Enter a statement or claim to verify:")

# ----------------------------------------
# Process Query
# ----------------------------------------
if user_query:
    # Clean and encode the query safely
    clean_query = (
        user_query.strip()
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("‘", "'")
        .replace("’", "'")
    )
    encoded_query = urllib.parse.quote(clean_query)

    if not clean_query:
        st.warning("⚠️ Please enter a valid statement to search.")
        st.stop()

    st.info(f"🔎 Searching fact-checks for: **{clean_query}** ...")

    api_url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={encoded_query}&key={API_KEY}"

    try:
        # Send API request
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Check if data contains claims
        if "claims" in data and data["claims"]:
            results = []
            for claim in data["claims"]:
                text = claim.get("text", "N/A")
                claimant = claim.get("claimant", "Unknown")
                date = claim.get("claimDate", "N/A")

                for review in claim.get("claimReview", []):
                    publisher = review.get("publisher", {}).get("name", "N/A")
                    title = review.get("title", "N/A")
                    url = review.get("url", "N/A")
                    rating = review.get("textualRating", "N/A")

                    results.append({
                        "Statement": text,
                        "Claimant": claimant,
                        "Date": date,
                        "Source": publisher,
                        "Review Title": title,
                        "Rating": rating,
                        "URL": url
                    })

            # Convert to DataFrame
            if results:
                df = pd.DataFrame(results)
                st.success(f"✅ Found {len(df)} results.")
                st.dataframe(df, use_container_width=True)

                # Comparison summary
                st.subheader("📊 Comparison Summary (by Source)")
                summary = (
                    df.groupby("Source")["Rating"]
                    .apply(lambda x: x.mode()[0] if not x.mode().empty else "N/A")
                    .reset_index()
                )
                st.table(summary)
            else:
                st.warning("⚠️ No detailed reviews found for this statement.")

        else:
            st.warning("⚠️ No fact-check results found for this query.")

    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ HTTP Error: {http_err}")
        st.write("API Response:", response.text)
    except Exception as e:
        st.error(f"⚠️ Unexpected Error: {e}")
