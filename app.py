import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Load BERT Sentiment Model
sentiment_pipeline = pipeline("sentiment-analysis")

# Streamlit UI
st.title("NYTimes Comments vs. Articles Sentiment Analysis (BERT)")
st.write("Upload CSV files for NYT articles and comments to analyze sentiment trends.")

# File Uploaders
comments_file = st.file_uploader("Upload Comments CSV", type=["csv"])
articles_file = st.file_uploader("Upload Articles CSV", type=["csv"])

def process_articles(df):
    """Combine relevant columns into a single 'article_text' field."""
    text_cols = ["abstract", "snippet", "lead_paragraph"]
    df["article_text"] = df[text_cols].fillna('').agg(' '.join, axis=1)  # Merge available columns
    return df

def analyze_sentiment(df, text_column):
    """Apply BERT sentiment analysis to a text column."""
    def get_sentiment(text):
        result = sentiment_pipeline(text)[0]  # Run sentiment analysis once
        return pd.Series([result["label"], result["score"]])

    df[["sentiment", "score"]] = df[text_column].astype(str).apply(get_sentiment)
    return df

if comments_file:
    df_comments = pd.read_csv(comments_file)
    if "commentBody" in df_comments.columns:
        st.write("Processing Comments...")
        df_comments = analyze_sentiment(df_comments, "commentBody")
        st.dataframe(df_comments[["commentBody", "sentiment", "score"]])  # Show results
    else:
        st.error("CSV file must contain a 'commentBody' column.")

if articles_file:
    df_articles = pd.read_csv(articles_file)
    if any(col in df_articles.columns for col in ["abstract", "snippet", "lead_paragraph"]):
        st.write("Processing Articles...")
        df_articles = process_articles(df_articles)  # Merge columns
        df_articles = analyze_sentiment(df_articles, "article_text")
        st.dataframe(df_articles[["article_text", "sentiment", "score"]])  # Show results
    else:
        st.error("CSV file must contain 'abstract', 'snippet', or 'lead_paragraph'.")

# If sentiment analysis is done, proceed with visualizations
if "sentiment" in df_comments.columns and "sentiment" in df_articles.columns:
    
    # Sentiment Distribution Visualization
    st.subheader("Sentiment Distribution (Articles vs. Comments)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    df_articles["sentiment"].value_counts().plot(kind="bar", ax=ax[0], title="NYT Articles Sentiment", color=['green', 'red', 'blue'])
    df_comments["sentiment"].value_counts().plot(kind="bar", ax=ax[1], title="NYT Comments Sentiment", color=['green', 'red', 'blue'])
    
    st.pyplot(fig)
    
    # Sentiment Over Time Visualization
    st.subheader("Sentiment Trends Over Time")
    
    if "commentDate" in df_comments.columns and "articleDate" in df_articles.columns:
        df_comments["date"] = pd.to_datetime(df_comments["commentDate"], errors='coerce')
        df_articles["date"] = pd.to_datetime(df_articles["articleDate"], errors='coerce')
        
        df_comments_time = df_comments.groupby("date")["score"].mean()
        df_articles_time = df_articles.groupby("date")["score"].mean()

        plt.figure(figsize=(12, 5))
        plt.plot(df_comments_time, label="Comments Sentiment", color="red")
        plt.plot(df_articles_time, label="Articles Sentiment", color="blue")
        plt.xlabel("Date")
        plt.ylabel("Average Sentiment Score")
        plt.title("Sentiment Trends Over Time")
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Date columns not found. Skipping sentiment trend analysis.")

    # Sentiment by Topic
    st.subheader("Sentiment by Topic")
    keywords = ["China", "Trade War", "Steel Tariff", "Protectionism"]
    for keyword in keywords:
        df_topic = df_comments[df_comments["commentBody"].str.contains(keyword, case=False, na=False)]
        if not df_topic.empty:
            avg_sentiment = df_topic["score"].mean()
            st.write(f"**Average sentiment for '{keyword}':** {avg_sentiment:.3f}")
        else:
            st.write(f"No comments found for topic: {keyword}")

    # Most Positive & Negative Comments
    st.subheader("Most Positive & Negative Comments")
    
    st.write("**Most Positive Comments:**")
    st.write(df_comments.nlargest(5, "score")[["commentBody", "score"]])
    
    st.write("**Most Negative Comments:**")
    st.write(df_comments.nsmallest(5, "score")[["commentBody", "score"]])
    
    # Download processed CSVs
    st.subheader("Download Sentiment-Tagged Data")

    # Convert DataFrames to CSVs in memory
    comments_csv = df_comments.to_csv(index=False).encode('utf-8')
    articles_csv = df_articles.to_csv(index=False).encode('utf-8')

    st.download_button("Download Comments CSV", data=comments_csv, file_name="nyt_comments_with_sentiment.csv", mime="text/csv")
    st.download_button("Download Articles CSV", data=articles_csv, file_name="nyt_articles_with_sentiment.csv", mime="text/csv")
