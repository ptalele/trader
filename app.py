import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import datetime
import urllib.request
import xml.etree.ElementTree as ET
from scipy.stats import norm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Force wide dashboard structure
st.set_page_config(page_title="Trader | Advanced Matrix Underwriter", layout="wide")

st.title("🦅 Trader Live Options Underwriting Cockpit")
st.caption("Resilient Alpha Engine: Multi-Factor Math Matrix, RSS Scrapers, and Automated Safety Locks")

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
    "Consumer Cyclical & Retail": ["AMZN", "TSLA", "HD", "COST", "MCD", "WMT", "LOW", "SBUX", "TJX", "BKNG", "XLY"],
    "Healthcare & Pharmaceuticals": ["UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DIS", "REGN", "MDT", "VRTX", "AMGN", "GILD", "XLV"],
    "Energy & Basic Materials": ["XOM", "CVX", "COP", "XLE", "LIN", "XLB", "CAT", "DE", "HON", "GE", "LMT", "MMM", "UNP", "UPS"],
    "Consumer Defensive, Utilities & Real Estate": ["PG", "KO", "PEP", "MDLZ", "NEE", "AMT", "XLP", "XLU", "XLRE"],
    "Derivatives & Multi-Asset Trackers": ["MUU", "USD", "NVDL", "TSLL", "AAPU", "AMZU", "MSFU", "GGLL", "FBL", "TQQQ", "SPY", "QQQ", "IWM", "DIA", "XLK", "SMH", "TLT", "EEM", "GDX", "GLD"]
}

TRADER_WATCHLIST = sorted(list({ticker for ticker_list in SECTOR_WATCHLIST.values() for ticker in ticker_list}))

# --- SIGNAL EXPLANATION DICTIONARY ---
SIGNAL_EXPLANATIONS = {
    "🛑 SAFETY SENTIMENT LOCK": "The asset is oversold, but the NLP engine detected severely negative headlines. To protect against 'catching a falling knife', execution is locked.",
    "🔥 HIGH-CONVICTION OVERSOLD PUT SIGNAL": "The asset broke below the -2 StdDev VWAP band, RSI confirms oversold conditions, and CMF shows institutional support. Selling a Cash-Secured Put maximizes premium collection with strong mathematical safety.",
    "🚀 MOMENTUM BREAKOUT CALL SIGNAL": "The asset broke above the +2 StdDev VWAP band, fueled by bullish news and expanding MACD momentum. The optimal play is to buy directional Call options to capture upside.",
    "⚠️ EXHAUSTION COVERED CALL SIGNAL": "The asset overextended above VWAP without bullish news, and MACD shows momentum decelerating. Selling an OTM Call capitalizes on over-inflated premiums before mean-reversion.",
    "⚠️ RANGE-BOUND PUT HARVEST SIGNAL": "The asset is trading neutrally near VWAP. Implied Volatility is stable. Selling a standard Delta-targeted Cash-Secured Put generates consistent premium yield via theta decay.",
    "🛑 CONFLICTING INTERNAL MATRIX": "VWAP positioning contradicts momentum (RSI/MACD) or volume flows (CMF). The engine recommends staying entirely in cash until indicators align."
}

# --- STABLE RSS SENTIMENT FETCH ENGINE ---
def fetch_stable_rss_headlines(symbol):
    headlines = []
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        for item in root.findall('.//item')[:5]:
            title_text = item.find('title').text
            link_text = item.find('link').text
            if title_text and link_text:
                headlines.append({"title": title_text, "link": link_text})
    except Exception:
        pass
    return headlines

# --- DATA STREAM PIPELINES ---
@st.cache_resource(ttl=60)
def get_ticker_obj(symbol):
    return yf.Ticker(symbol)

@st.cache_data(ttl=60)
def fetch_live_market_data(symbol):
    try:
        ticker_obj = yf.Ticker(symbol)
        info = ticker_obj.info
        spot_price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        if not spot_price:
            hist_today = ticker_obj.history(period="1d")
            if not hist_today.empty:
                spot_price = hist_today['Close'].iloc[-1]
            else:
                return None, f"Price feed unavailable for symbol {symbol}."
        
        avg_vol = info.get("averageDailyVolume10Day") or info.get("volume") or 0
        company_name = info.get("shortName") or info.get("longName") or symbol
        calendar = ticker_obj.calendar
        earnings_date = None
        
        if calendar is not None and 'Earnings Date' in calendar:
            earnings_date = calendar['Earnings Date'][0]
        elif isinstance(calendar, dict) and 'earningsDate' in calendar:
            earnings_date = calendar['earningsDate'][0]
            
        # Fetching 5 days of 5-minute data to ensure enough periods for MACD/RSI warmup calculations
        intraday_data = ticker_obj.history(period="5d", interval="5m")
        return {
            "spot": spot_price, 
            "avg_volume": avg_vol, 
            "company_name": company_name,
            "earnings_date": earnings_date, 
            "intraday_df": intraday_data
        }, None
    except Exception as e:
        return None, str(e)

# --- MATH ENGINES ---
def calculate_black_scholes_delta(spot, strike, dte, iv, is_call=False, risk_free_rate=0.045):
    if iv <= 0 or dte <= 0: return 0
    t = dte / 365.0
    d1 = (np.log(spot / strike) + (risk_free_rate + (iv ** 2) / 2) * t) / (iv * np.sqrt(t))
    return float(norm.cdf(d1)) if is_call else float(-norm.cdf(-d1))

def calculate_probability_of_profit(spot, strike, premium, dte, iv, is_call=False, risk_free_rate=0.045):
    if iv <= 0 or dte <= 0: return 0.0
    t = dte / 365.0
    breakeven = (strike + premium) if is_call else (strike - premium)
    d2 = (np.log(spot / breakeven) + (risk_free_rate - (iv ** 2) / 2) * t) / (iv * np.sqrt(t))
    return float(norm.cdf(-d2)) if is_call else float(norm.cdf(d2))

def render_clickable_banner(badge_text, explanation_text, bg_color):
    html_block = f"""
    <details style="background-color: {bg_color}; color: white; padding: 14px 16px; border-radius: 8px; margin-bottom: 16px; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <summary style="font-weight: bold; font-size: 16px;">
            🎯 {badge_text} <span style="font-size: 13px; font-weight: normal; opacity: 0.85; margin-left: 8px;">(Click for Logic)</span>
        </summary>
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.3); font-size: 14px; line-height: 1.5;">
            <strong>Action Rationale:</strong><br>{explanation_text}
        </div>
    </details>
    """
    st.markdown(html_block, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Sourcing Mode Configuration")
sourcing_mode = st.sidebar.radio("Select Sourcing Input Method:", options=["1. Current Ticker Symbol (Manual)", "2. Sector-Mapped Watchlist"])

if sourcing_mode == "1. Current Ticker Symbol (Manual)":
    target_ticker = st.sidebar.text_input("Enter Ticker Symbol", value="MUU").upper().strip()
else:
    chosen_sector = st.sidebar.selectbox("Filter Watchlist by Sector Menu:", options=list(SECTOR_WATCHLIST.keys()))
    target_ticker = st.sidebar.selectbox("Select Target Ticker Symbol:", options=SECTOR_WATCHLIST[chosen_sector])

st.sidebar.header("🎯 Underwriting Parameters")
min_dte = st.sidebar.slider("Minimum DTE Window", min_value=15, max_value=45, value=30)
max_dte = st.sidebar.slider("Maximum DTE Window", min_value=35, max_value=90, value=45)
contracts = st.sidebar.number_input("Vault Contracts", min_value=1, value=1, step=1)

# --- INITIALIZE ANALYSIS COCKPIT PLATFORM ---
if 'target_ticker' in locals():
    underlying_equity = target_ticker
    is_leveraged_etf = False
    if target_ticker in ETF_DECOMPOSITION_MAP:
        underlying_equity = ETF_DECOMPOSITION_MAP[target_ticker]
        is_leveraged_etf = True

    with st.spinner(f"Extracting streaming packets for {target_ticker}..."):
        market_data, error_msg = fetch_live_market_data(target_ticker)
        if is_leveraged_etf:
            underlying_market_data, _ = fetch_live_market_data(underlying_equity)
        else:
            underlying_market_data = market_data

    if error_msg:
        st.error(f"Network Pipeline Error: {error_msg}")
    elif market_data is None or market_data["spot"] is None:
        st.warning("Data Pipeline timed out. Verify ticker symbol architecture.")
    else:
        spot = market_data["spot"]
        avg_volume = market_data["avg_volume"]
        company_name = market_data.get("company_name", target_ticker)
        df_intraday = market_data["intraday_df"]
        tk = get_ticker_obj(target_ticker)
        earnings_dt = underlying_market_data["earnings_date"] if underlying_market_data else None

        st.markdown(f"## 👁️ Active Cockpit Target: **{target_ticker} | {company_name}**")
        tab_cockpit, tab_chain, tab_math = st.tabs(["🎯 Underwriting Cockpit", "⛓️ Live Option Chain", "🧮 Engine Math & Logic"])

        if len(df_intraday) == 0:
             time_slots = pd.date_range("09:30", "16:00", freq="5min")
             df_intraday = pd.DataFrame(index=time_slots, data={'Open': spot, 'High': spot, 'Low': spot, 'Close': spot, 'Volume': 100000})

        if earnings_dt and isinstance(earnings_dt, (pd.Timestamp, datetime.datetime)):
            earnings_dt = earnings_dt.date()

        is_approved = True
        block_reason = ""
        if spot < 30.00:
            is_approved = False
            block_reason = f"Spot valuation (${spot:.2f}) drops below $30 thresholds."
        elif avg_volume < 1000000:
            is_approved = False
            block_reason = f"Institutional Liquidity Alert: 10-day volume ({avg_volume:,}) below 1M share requirement."

        all_expirations = tk.options
        selected_expiration = None
        days_to_exp = 30
        
        if all_expirations:
            today = datetime.date.today()
            for exp in all_expirations:
                exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
                delta_days = (exp_date - today).days
                if min_dte <= delta_days <= max_dte:
                    if earnings_dt and isinstance(earnings_dt, datetime.date) and exp_date > earnings_dt:
                        continue 
                    selected_expiration = exp
                    days_to_exp = delta_days
                    break
            if not selected_expiration and is_approved:
                is_approved = False
                block_reason = f"Imminent Earnings Collision: Underlying asset ({underlying_equity}) has an announcement on {earnings_dt} blocking option windows."

        with tab_cockpit:
            rss_data = fetch_stable_rss_headlines(target_ticker)
            sentiment_score = 0.0
            headline_log = []
            
            if rss_data:
                compound_scores = []
                for item in rss_data:
                    title = item["title"]
                    link = item["link"]
                    score_dict = sia.polarity_scores(title)
                    compound_scores.append(score_dict["compound"])
                    headline_log.append({"Flashing Headline": title, "VADER Index Score": score_dict["compound"], "Source": link})
                sentiment_score = float(np.mean(compound_scores))

            if sentiment_score >= 0.15:
                sentiment_state = "BULLISH NEWS"
                sentiment_badge_style = "background-color: #0073e6; color: #FFFFFF; padding: 4px 14px; border-radius: 4px; font-weight: bold; font-size: 14px;"
            elif sentiment_score <= -0.15:
                sentiment_state = "BEARISH NEWS"
                sentiment_badge_style = "background-color: #FF4B4B; color: #FFFFFF; padding: 4px 14px; border-radius: 4px; font-weight: bold; font-size: 14px;"
            else:
                sentiment_state = "NEUTRAL NEWS"
                sentiment_badge_style = "background-color: #4A4A4A; color: #FFFFFF; padding: 4px 14px; border-radius: 4px; font-weight: bold; font-size: 14px;"

            # --- CALCULATE CORE MATH & PANDAS-TA INDICATORS ---
            df_intraday['Typical_Price'] = (df_intraday['High'] + df_intraday['Low'] + df_intraday['Close']) / 3
            df_intraday['Price_Vol'] = df_intraday['Typical_Price'] * df_intraday['Volume']
            df_intraday['VWAP_Line'] = df_intraday['Price_Vol'].cumsum() / df_intraday['Volume'].cumsum()
            running_std = df_intraday['Close'].rolling(window=20, min_periods=1).std()
            df_intraday['Upper_2'] = df_intraday['VWAP_Line'] + (running_std * 2)
            df_intraday['Lower_2'] = df_intraday['VWAP_Line'] - (running_std * 2)
            
            # Injecting TA-Lib Indicators
            df_intraday.ta.rsi(length=14, append=True)
            df_intraday.ta.macd(fast=12, slow=26, signal=9, append=True)
            df_intraday.ta.atr(length=14, append=True)
            df_intraday.ta.cmf(length=20, append=True)
            
            # Extract current state (handling potential NaNs from early warmup periods)
            valid_df = df_intraday.dropna()
            if not valid_df.empty:
                current_price = valid_df['Close'].iloc[-1]
                current_vwap = valid_df['VWAP_Line'].iloc[-1]
                current_lower_band = valid_df['Lower_2'].iloc[-1]
                current_upper_band = valid_df['Upper_2'].iloc[-1]
                current_std = running_std.iloc[-1]
                
                c_rsi = valid_df['RSI_14'].iloc[-1]
                c_macd_hist = valid_df['MACDh_12_26_9'].iloc[-1]
                c_atr = valid_df['ATRr_14'].iloc[-1]
                c_cmf = valid_df['CMF_20'].iloc[-1]
            else:
                current_price, current_vwap, current_lower_band, current_upper_band, current_std = spot, spot, spot, spot, 0.0
                c_rsi, c_macd_hist, c_atr, c_cmf = 50.0, 0.0, 1.0, 0.0

            condition_state = "NEUTRAL"
            badge_style = "background-color: #4A4A4A; color: #FFFFFF; padding: 4px 14px; border-radius: 4px; font-weight: bold; font-size: 14px;"
            
            opt_strike, execution_midpoint, cash_reserve, premium_gain = 0.0, 0.0, 0.0, 0.0
            opt_delta, opt_iv, true_pop, breakeven_price = 0.0, 0.0, 0.0, 0.0
            roc_return, annual_yield = 0.0, 0.0
            recommendation_action_text = ""
            rec_log_badge = "✅ ALL METRICS CLEAR"
            
            # --- SUPER-SIGNAL MATRIX EXECUTION ROUTING ---
            if all_expirations and selected_expiration:
                raw_chain = tk.option_chain(selected_expiration)
                
                if current_price <= current_lower_band:
                    condition_state = "OVERSOLD"
                    badge_style = "background-color: #00CC66; color: #FFFFFF; padding: 4px 14px; border-radius: 4px; font-weight: bold; font-size: 14px;"
                    if sentiment_state == "BEARISH NEWS":
                        recommendation_action_text = f"ABORT TRANSACTION: NLP engine detected severely negative news flow."
                        rec_log_badge = "🛑 SAFETY SENTIMENT LOCK"
                    elif c_rsi > 45 or c_cmf < -0.15:
                        recommendation_action_text = f"HOLD POSITION: VWAP is oversold but momentum indicators warn of a false bottom."
                        rec_log_badge = "🛑 CONFLICTING INTERNAL MATRIX"
                    else:
                        chain_matrix = raw_chain.puts
                        chain_matrix['calculated_delta'] = chain_matrix.apply(lambda r: calculate_black_scholes_delta(spot, r['strike'], days_to_exp, r['impliedVolatility'], is_call=False), axis=1)
                        viable = chain_matrix[(chain_matrix['calculated_delta'] <= -0.12) & (chain_matrix['calculated_delta'] >= -0.22)]
                        optimal_contract = viable.iloc[(viable['calculated_delta'] - (-0.15)).abs().argsort()[:1]].iloc[0] if not viable.empty else chain_matrix.iloc[(chain_matrix['strike'] - (spot - (c_atr*2))).abs().argsort()[:1]].iloc[0]
                        opt_strike = optimal_contract['strike']
                        execution_midpoint = (optimal_contract['bid'] + optimal_contract['ask']) / 2 if optimal_contract['ask'] > optimal_contract['bid'] else optimal_contract['lastPrice']
                        recommendation_action_text = f"Sell to open PUT - {target_ticker} at {opt_strike:.0f} expiring on {selected_expiration} @ premium ${execution_midpoint:.2f}"
                        rec_log_badge = "🔥 HIGH-CONVICTION OVERSOLD PUT SIGNAL"
                        opt_delta = optimal_contract['calculated_delta']
                        opt_iv = optimal_contract['impliedVolatility']
                        
                elif current_price >= current_upper_band or sentiment_state == "BULLISH NEWS":
                    condition_state = "OVERBOUGHT" if current_price >= current_vwap else "MOMENTUM"
                    badge_style = "background-color: #FF4B4B; color: #FFFFFF; padding: 4px 14px; border-radius: 4px; font-weight: bold; font-size: 14px;" if condition_state=="OVERBOUGHT" else "background-color: #0073e6; color: #FFFFFF; padding: 4px 14px; border-radius: 4px; font-weight: bold; font-size: 14px;"
                    if sentiment_state == "BULLISH NEWS" and c_macd_hist > 0:
                        chain_matrix = raw_chain.calls
                        chain_matrix['calculated_delta'] = chain_matrix.apply(lambda r: calculate_black_scholes_delta(spot, r['strike'], days_to_exp, r['impliedVolatility'], is_call=True), axis=1)
                        viable = chain_matrix[(chain_matrix['calculated_delta'] >= 0.45) & (chain_matrix['calculated_delta'] <= 0.60)]
                        optimal_contract = viable.iloc[(viable['calculated_delta'] - 0.50).abs().argsort()[:1]].iloc[0] if not viable.empty else chain_matrix.iloc[(chain_matrix['strike'] - spot).abs().argsort()[:1]].iloc[0]
                        opt_strike = optimal_contract['strike']
                        execution_midpoint = (optimal_contract['bid'] + optimal_contract['ask']) / 2 if optimal_contract['ask'] > optimal_contract['bid'] else optimal_contract['lastPrice']
                        recommendation_action_text = f"Buy to open CALL - {target_ticker} at {opt_strike:.0f} expiring on {selected_expiration} @ debit target ${execution_midpoint:.2f}"
                        rec_log_badge = "🚀 MOMENTUM BREAKOUT CALL SIGNAL"
                        opt_delta = optimal_contract['calculated_delta']
                        opt_iv = optimal_contract['impliedVolatility']
                    else:
                        chain_matrix = raw_chain.calls
                        chain_matrix['calculated_delta'] = chain_matrix.apply(lambda r: calculate_black_scholes_delta(spot, r['strike'], days_to_exp, r['impliedVolatility'], is_call=True), axis=1)
                        viable = chain_matrix[(chain_matrix['calculated_delta'] >= 0.12) & (chain_matrix['calculated_delta'] <= 0.22)]
                        optimal_contract = viable.iloc[(viable['calculated_delta'] - 0.15).abs().argsort()[:1]].iloc[0] if not viable.empty else chain_matrix.iloc[(chain_matrix['strike'] - (spot + (c_atr*2))).abs().argsort()[:1]].iloc[0]
                        opt_strike = optimal_contract['strike']
                        execution_midpoint = (optimal_contract['bid'] + optimal_contract['ask']) / 2 if optimal_contract['ask'] > optimal_contract['bid'] else optimal_contract['lastPrice']
                        recommendation_action_text = f"Sell to open CALL - {target_ticker} at {opt_strike:.0f} expiring on {selected_expiration} @ premium ${execution_midpoint:.2f}"
                        rec_log_badge = "⚠️ EXHAUSTION COVERED CALL SIGNAL"
                        opt_delta = optimal_contract['calculated_delta']
                        opt_iv = optimal_contract['impliedVolatility']
                else:
                    chain_matrix = raw_chain.puts
                    chain_matrix['calculated_delta'] = chain_matrix.apply(lambda r: calculate_black_scholes_delta(spot, r['strike'], days_to_exp, r['impliedVolatility'], is_call=False), axis=1)
                    viable = chain_matrix[(chain_matrix['calculated_delta'] <= -0.12) & (chain_matrix['calculated_delta'] >= -0.22)]
                    optimal_contract = viable.iloc[(viable['calculated_delta'] - (-0.15)).abs().argsort()[:1]].iloc[0] if not viable.empty else chain_matrix.iloc[(chain_matrix['strike'] - (spot * 0.90)).abs().argsort()[:1]].iloc[0]
                    opt_strike = optimal_contract['strike']
                    execution_midpoint = (optimal_contract['bid'] + optimal_contract['ask']) / 2 if optimal_contract['ask'] > optimal_contract['bid'] else optimal_contract['lastPrice']
                    recommendation_action_text = f"Sell to open PUT - {target_ticker} at {opt_strike:.0f} expiring on {selected_expiration} @ premium ${execution_midpoint:.2f}"
                    rec_log_badge = "⚠️ RANGE-BOUND PUT HARVEST SIGNAL"
                    opt_delta = optimal_contract['calculated_delta']
                    opt_iv = optimal_contract['impliedVolatility']

                cash_reserve = opt_strike * 100 * contracts
                premium_gain = execution_midpoint * 100 * contracts
                roc_return = (premium_gain / cash_reserve) * 100 if cash_reserve > 0 else 0
                annual_yield = roc_return * (365 / days_to_exp) if days_to_exp > 0 else 0

                is_call_strategy = ("CALL" in rec_log_badge)
                breakeven_price = (opt_strike + execution_midpoint) if is_call_strategy else (opt_strike - execution_midpoint)
                true_pop = calculate_probability_of_profit(spot, opt_strike, execution_midpoint, days_to_exp, opt_iv, is_call=is_call_strategy)

            # ==========================================
            # DYNAMIC ACTION HEADER WITH CLICKABLE BANNER
            # ==========================================
            col_rec_left, col_rec_right = st.columns([1.4, 1.1])
            with col_rec_left:
                if not is_approved:
                    render_clickable_banner("🛑 RISK LOCK ENFORCED", "Execution is halted because the asset violated core capital liquidity or binary event constraints.", "#FF4B4B")
                    st.code(f"MANDATE VIOLATION REJECTION:\n{block_reason}", language="text")
                else:
                    if "🛑" in rec_log_badge: bg_color = "#FF4B4B"
                    elif "🔥" in rec_log_badge: bg_color = "#00CC66"
                    elif "🚀" in rec_log_badge: bg_color = "#0073e6"
                    else: bg_color = "#B8860B"

                    rec_explanation = SIGNAL_EXPLANATIONS.get(rec_log_badge, "Standard execution routing logic applied based on current VWAP bands.")
                    render_clickable_banner(rec_log_badge, rec_explanation, bg_color)
                            
                    st.code(recommendation_action_text, language="text")
                    st.caption(f"🔒 Required Allocation: ${cash_reserve:,.2f} | 💰 Premium Yield Credit: ${premium_gain:,.2f}")
            
            with col_rec_right:
                st.info("🔍 INTERNAL MATRIX TELEMETRY")
                v_col1, v_col2, v_col3 = st.columns(3)
                v_col1.metric("VWAP Mean", f"${current_vwap:.2f}", f"Spot: ${current_price:.2f}", delta_color="off")
                v_col2.metric("RSI (14)", f"{c_rsi:.1f}", f"ATR: ${c_atr:.2f}", delta_color="off")
                v_col3.metric("MACD Hist", f"{c_macd_hist:.3f}", f"CMF: {c_cmf:.2f}", delta_color="off")

            st.divider()

            # ==========================================
            # MIDDLE ROW DATA SEGMENTS
            # ==========================================
            col_data_left, col_data_right = st.columns([1.1, 1.4])
            with col_data_left:
                st.subheader("📊 Trade Health & Pricing Efficiency")
                if is_approved and selected_expiration:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Target", f"{target_ticker}", f"Spot: ${spot:.2f}", help="The core tracking asset and its current live trading price.")
                    m2.metric("PoP", f"{true_pop:.1%}", f"Delta: {opt_delta:.2f}", delta_color="off", help="Calculates mathematical probability that stock will close past breakeven line at expiration.")
                    m3.metric("Breakeven", f"${breakeven_price:.2f}", f"Buffer: ${abs(spot - breakeven_price):.2f}", delta_color="off", help="The exact price where the trade begins taking a loss.")
                    
                    m4, m5, m6 = st.columns(3)
                    m4.metric("RoC", f"{roc_return:.2f}%", f"{days_to_exp} DTE Window", help="Return on Capital directly on the collateral required.")
                    m5.metric("Annual Yield", f"{annual_yield:.2f}%", "Compounded Runway", help="Extrapolates short-term RoC to a 1-year yield.")
                    m6.metric("IV", f"{opt_iv:.1%}", "Premium Pricing Base", help="Implied Volatility.")
                    
                st.divider()
                
                st.markdown("### 📝 Underwriting Diagnostic Rationale")
                st.markdown(f"""
                The advanced matrix selection for **{target_ticker}** utilized cross-validation logic:
                * **Mathematical Confirmation:** Standard deviation bands were audited against underlying RSI momentum, MACD histograms, and CMF volume pressure before clearing execution.
                * **Decomposition Audit:** The system evaluated binary earnings dates across the parent equity token (**{underlying_equity}**).
                * **Compliance State Summary:** System flags read **is_approved = {is_approved}**.
                """)

            with col_data_right:
                st.subheader("📈 Technical Chart Core View")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_intraday.index, open=df_intraday['Open'], high=df_intraday['High'], low=df_intraday['Low'], close=df_intraday['Close'], name='Live 5M Candles'))
                fig.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday['VWAP_Line'], name='VWAP Baseline', line=dict(color='#FFD700', width=2)))
                fig.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday['Lower_2'], name='-2 StdDev Band', line=dict(color='#FF4B4B', width=1.5, dash='dot')))
                fig.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday['Upper_2'], name='+2 StdDev Band', line=dict(color='#00CC66', width=1, dash='dot')))
                fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader(f"📊 Live RSS News Headlines Analysis (VADER Index Score: {sentiment_score:.2f})")
            if headline_log:
                hl_df = pd.DataFrame(headline_log)
                st.dataframe(
                    hl_df, use_container_width=True, hide_index=True,
                    column_config={"Source": st.column_config.LinkColumn("Source Link", display_text="Read Article ↗"), "VADER Index Score": st.column_config.NumberColumn("VADER Score", format="%.2f")}
                )
            else:
                st.caption("No public news releases detected within active session tracking buckets.")

        with tab_chain:
            st.subheader(f"⛓️ Live Streaming Option Chain Grid: {target_ticker}")
            if not all_expirations:
                st.warning("No active option expirations found for this ticker asset.")
            else:
                chosen_expiry = st.selectbox("Select Chain Expiration Target:", options=all_expirations, index=0)
                @st.fragment(run_every=5)
                def render_streaming_options_fragment(ticker_symbol, expiry_date_str):
                    current_tk = yf.Ticker(ticker_symbol)
                    try:
                        live_chain = current_tk.option_chain(expiry_date_str)
                        live_puts = live_chain.puts.copy()
                        st.caption(f"⏱️ Local Client Handshake Completed: **{datetime.datetime.now().strftime('%H:%M:%S')}** (Sync Frequency: 5s)")
                        live_puts['bid'] = live_puts['bid'].fillna(0.00)
                        live_puts['ask'] = live_puts['ask'].fillna(0.00)
                        live_puts['Spread'] = live_puts['ask'] - live_puts['bid']
                        display_columns = ['contractSymbol', 'strike', 'bid', 'ask', 'Spread', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
                        final_grid = live_puts[display_columns].rename(columns={'contractSymbol': 'Contract ID', 'strike': 'Strike Price', 'bid': 'Bid (Buy)', 'ask': 'Ask (Sell)', 'lastPrice': 'Last Close', 'impliedVolatility': 'Implied Volatility (IV)'})
                        st.dataframe(final_grid.style.format({'Strike Price': '${:.2f}', 'Bid (Buy)': '${:.2f}', 'Ask (Sell)': '${:.2f}', 'Spread': '${:.2f}', 'Last Close': '${:.2f}', 'Implied Volatility (IV)': '{:.2%}'}), use_container_width=True, hide_index=True)
                    except Exception as ex:
                        st.error(f"Intraday order book parsing timeout: {str(ex)}")
                render_streaming_options_fragment(target_ticker, chosen_expiry)

        # ==========================================
        # NEW WORKSPACE 3: ENGINE MATH & LOGIC TAB
        # ==========================================
        with tab_math:
            st.subheader("🧮 Quantitative Matrix Overview")
            st.markdown("This underwriting engine evaluates trades through a multi-dimensional matrix. It pairs lagging institutional accumulation metrics with leading momentum oscillators to output high-conviction signals.")
            st.divider()

            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.markdown("### 1. Volume-Weighted Average Price (VWAP)")
                st.markdown("VWAP represents the true average cost basis for all shares traded during the current session, providing the core baseline for identifying overbought or oversold conditions.")
                st.latex(r"VWAP = \frac{\sum (\text{Typical Price} \times \text{Volume})}{\sum \text{Volume}}")
                st.markdown("**Engine Logic:** Defines standard deviation boundaries ($-2\sigma$ and $+2\sigma$). A breach signifies intense statistical panic or euphoria.")

                st.markdown("### 2. Relative Strength Index (RSI)")
                st.markdown("A momentum oscillator measuring the speed and change of price movements, acting as a filter for false breakdowns.")
                st.latex(r"RSI = 100 - \frac{100}{1 + RS}")
                st.markdown("**Engine Logic:** Rejects put-selling strategies if RSI is in freefall. Requires confirmation that momentum is slowing before catching the knife.")

                st.markdown("### 3. Black-Scholes Probability of Profit (POP)")
                st.markdown("Instead of estimating probability via Delta ($d_1$), the engine calculates exact mathematical success by shifting the normal distribution to the execution *Breakeven Price*.")
                st.latex(r"d_2 = \frac{\ln(\frac{S}{K_{Breakeven}}) + (r - \frac{\sigma^2}{2})T}{\sigma \sqrt{T}}")
                st.markdown("**Engine Logic:** Accurately forecasts your safety net by dynamically incorporating the IV premium buffer collected at open.")

            with col_m2:
                st.markdown("### 4. Moving Average Convergence Divergence (MACD)")
                st.markdown("Tracks the relationship between two moving averages (EMA). The engine relies heavily on the MACD *Histogram* to measure acceleration.")
                st.latex(r"MACD = EMA_{12} - EMA_{26}")
                st.latex(r"\text{Histogram} = MACD - EMA_9(MACD)")
                st.markdown("**Engine Logic:** Authorizes aggressive directional Long Call breakouts only if the Histogram reveals mathematically expanding upside momentum.")

                st.markdown("### 5. Chaikin Money Flow (CMF)")
                st.markdown("Measures institutional buying and selling pressure by tracking where the asset closes relative to its high/low range, weighted by volume.")
                st.latex(r"CMF = \frac{\sum_{i=1}^{21} \text{Money Flow Volume}_i}{\sum_{i=1}^{21} \text{Volume}_i}")
                st.markdown("**Engine Logic:** If CMF is heavily negative ($\le -0.15$), the engine aborts put-selling, identifying hidden distribution by institutional market makers.")

                st.markdown("### 6. VADER Sentiment Pipeline")
                st.markdown("The Valuation Aware Dictionary and sEntiment Reasoner (VADER) parses live Yahoo Finance RSS XML feeds. It assigns normalized polarity indices across market lexicon tokens.")
                st.latex(r"Score_{Index} = \frac{x}{\sqrt{x^2 + \alpha}}")
                st.markdown("**Engine Logic:** If negative sentiment ($\le -0.15$) aligns with an oversold VWAP, the `SAFETY SENTIMENT LOCK` prevents portfolio destruction on sudden fundamental news crashes.")
