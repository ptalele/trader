import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import urllib.request
import xml.etree.ElementTree as ET
from scipy.stats import norm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Force wide dashboard structure
st.set_page_config(page_title="Trader | Advanced POP Underwriter", layout="wide")

st.title("🦅 Trader Live Options Underwriting Cockpit")
st.caption("Resilient Alpha Engine: POP Math, RSS Scrapers, Live Options Chains, and Automated Safety Locks")

# --- INITIALIZE NATURAL LANGUAGE SENTIMENT ANALYZER ---
@st.cache_resource
def load_sentiment_analyzer():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_sentiment_analyzer()

# --- LEVERAGED DERIVATIVE ETF TO CORE TICKER MAP ---
ETF_DECOMPOSITION_MAP = {
    "MUU": "MU", "USD": "NVDA", "NVDL": "NVDA", "TSLL": "TSLA",
    "AAPU": "AAPL", "AMZU": "AMZN", "MSFU": "MSFT", "GGLL": "GOOGL", "FBL": "META"
}

SECTOR_WATCHLIST = {
    "Technology & Semiconductors": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "AMD", "CRM", "CSCO", "TXN", "INTC", "QCOM", "AMAT", "LRCX", "ADI", "PANW", "MU", "ORCL", "IBM", "INTU"],
    "Communication Services & Internet": ["GOOGL", "META", "NFLX", "CMCSA", "VZ", "XLC", "PM"],
    "Financial Services & Banking": ["JPM", "BAC", "WFC", "GS", "MS", "AXP", "C", "XLF"],
    "Consumer Cyclical & Retail": ["AMZN", "TSLA", "HD", "COST", "MCD", "WMT", "LOW",
