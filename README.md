# 🦅 Trader Live Options Underwriting Cockpit

An institutional-grade multi-strategy trading engine built with **Streamlit** and **yfinance**. It features real-time option chain streaming (5s refresh state), direct Yahoo Finance RSS parsing integrated with **VADER NLP Sentiment analysis**, and automated non-blocking risk gates.

## ✨ Core Blueprint Architecture
* **Non-Blocking Risk Locks:** Renders complete underlying data matrix and option chains regardless of asset compliance state; explicitly enforces a warning block on the executable recommendations box if parameters drop below strict liquidity configurations.
* **RSS Sentiment Core Engine:** Evaluates rolling real-time headlines mapping arithmetic mean tokenization bounds directly through VADER NLP math vectors.
* **Leveraged ETF Tracker Decomposition:** Unwraps multi-asset tracker components down to parent equities to audit binary corporate earnings calendar constraints.

## 🚀 Quick Local Setup Instructions

### 1. Clone or Download Files
Ensure your project contains the following operational directory layout:
```text
├── app.py
├── requirements.txt
└── README.md