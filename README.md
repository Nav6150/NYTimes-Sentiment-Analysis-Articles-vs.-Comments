# NYTimes-Sentiment-Analysis-Articles-vs.-Comments
This project analyzes how sentiment differs between NYTimes articles and reader comments on the same topics. We focus on understanding tone, bias, and audience perception using Natural Language Processing (NLP).

🧠 What It Does
Scrapes and compiles NYTimes articles and their associated comments

Applies BERT-based sentiment analysis to both sources

Compares sentiment distribution and trends over time

Highlights topics with sentiment disparity (e.g. "China", "Trade War", "Tariffs")

Identifies the most positive and negative reader comments

Provides downloadable sentiment-tagged datasets

📦 Tech Stack
Python (pandas, matplotlib, Streamlit)

Hugging Face Transformers (pipeline for BERT sentiment)

Jupyter Notebooks for exploratory analysis

Streamlit app for interactive visualization and CSV upload

🚀 How to Run
Run app.py to launch the Streamlit dashboard.

Upload CSVs for NYTimes articles and comments.

Explore sentiment breakdowns and download results.

📁 Files
app.py: Streamlit app for interactive sentiment analysis

comments_nytimes.ipynb, NYT Analysis-1.ipynb: Notebooks for initial exploration and modeling

Step1_NYTimes_Scraper_Comments.ipynb: Web scraping and data preparation

currentarticles.csv, nytimes_comments.csv: Example datasets
