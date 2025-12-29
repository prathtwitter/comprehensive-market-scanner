import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Prath's Market Scanner", layout="wide", page_icon="🎯")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 20px; color: #00CC96; }
    .stRadio > div { flex-direction: row; } 
    .stDataFrame { border: 1px solid #333; }
    h3 { border-bottom: 2px solid #333; padding-bottom: 10px; }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        max-width: 100%;
    }
    div[data-testid="stDataFrame"] > div {
        overflow-x: auto;
    }
    .stDataFrame th {
        white-space: nowrap !important;
        font-size: 12px !important;
        padding: 4px 8px !important;
    }
    .stDataFrame td {
        text-align: center !important;
        padding: 4px 8px !important;
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SECURITY & AUTH
# ==========================================
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        if st.session_state["password"] == st.secrets.get("PASSWORD", "Sniper2025"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter Access Key", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter Access Key", type="password", on_change=password_entered, key="password"
        )
        st.error("⛔ Access Denied")
        return False
    else:
        return True

# ==========================================
# 3. SECTOR & TICKER DATA
# ==========================================

# US Sectors and their ETFs/tickers
US_SECTORS = {
    "📊 Sector & Index Overview": "us_sectors_indices.csv",
    "Technology": "us_technology.csv",
    "Healthcare": "us_healthcare.csv",
    "Financials": "us_financials.csv",
    "Consumer Discretionary": "us_consumer_discretionary.csv",
    "Consumer Staples": "us_consumer_staples.csv",
    "Energy": "us_energy.csv",
    "Industrials": "us_industrials.csv",
    "Materials": "us_materials.csv",
    "Utilities": "us_utilities.csv",
    "Real Estate": "us_real_estate.csv",
    "Communication Services": "us_communication.csv",
}

# India Sectors
INDIA_SECTORS = {
    "📊 Sector & Index Overview": "ind_sectors_indices.csv",
    "Nifty 50": "ind_nifty50.csv",
    "Nifty Bank": "ind_nifty_bank.csv",
    "Nifty IT": "ind_nifty_it.csv",
    "Nifty Pharma": "ind_nifty_pharma.csv",
    "Nifty Auto": "ind_nifty_auto.csv",
    "Nifty FMCG": "ind_nifty_fmcg.csv",
    "Nifty Metal": "ind_nifty_metal.csv",
    "Nifty Realty": "ind_nifty_realty.csv",
    "Nifty Energy": "ind_nifty_energy.csv",
    "Nifty Infra": "ind_nifty_infra.csv",
    "Nifty PSU Bank": "ind_nifty_psu_bank.csv",
    "Nifty Private Bank": "ind_nifty_private_bank.csv",
    "Nifty Media": "ind_nifty_media.csv",
    "Nifty Midcap 100": "ind_nifty_midcap100.csv",
    "Nifty Smallcap 100": "ind_nifty_smallcap100.csv",
}

@st.cache_data
def load_tickers_from_file(filename, is_india=False):
    """Load tickers from a CSV file."""
    if not os.path.exists(filename):
        st.error(f"❌ '{filename}' not found. Please ensure all data files are in the repository.")
        return [], {}
    
    try:
        df = pd.read_csv(filename)
        
        # Check if 'Ticker' and 'Name' columns exist
        if 'Ticker' in df.columns and 'Name' in df.columns:
            tickers = df['Ticker'].tolist()
            names = dict(zip(df['Ticker'], df['Name']))
        else:
            # Fallback: first column is ticker
            tickers = df.iloc[:, 0].tolist()
            names = {t: t for t in tickers}
        
        # Clean tickers
        clean_tickers = []
        clean_names = {}
        for t in tickers:
            t_clean = str(t).strip().upper()
            if is_india and not t_clean.endswith(".NS"):
                t_clean = f"{t_clean}.NS"
            clean_tickers.append(t_clean)
            # Map clean ticker to name
            original = str(t).strip().upper()
            if original in names:
                clean_names[t_clean] = names[original]
            else:
                clean_names[t_clean] = t_clean
        
        return clean_tickers, clean_names
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return [], {}

# ==========================================
# 4. MATH WIZ (Advanced Logic)
# ==========================================
class MathWiz:
    @staticmethod
    def calculate_choppiness(high, low, close, length=14):
        try:
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr_sum = tr.rolling(window=length).sum()
            high_max = high.rolling(window=length).max()
            low_min = low.rolling(window=length).min()
            
            range_diff = high_max - low_min
            range_diff.replace(0, np.nan, inplace=True)

            numerator = np.log10(atr_sum / range_diff)
            denominator = np.log10(length)
            return 100 * (numerator / denominator)
        except:
            return pd.Series(dtype='float64')

    @staticmethod
    def identify_strict_swings(df, neighbor_count=3):
        is_swing_high = pd.Series(True, index=df.index)
        is_swing_low = pd.Series(True, index=df.index)
        
        for i in range(1, neighbor_count + 1):
            is_swing_high &= (df['High'] > df['High'].shift(i))
            is_swing_low &= (df['Low'] < df['Low'].shift(i))
            is_swing_high &= (df['High'] > df['High'].shift(-i))
            is_swing_low &= (df['Low'] < df['Low'].shift(-i))
            
        return is_swing_high, is_swing_low

    @staticmethod
    def find_fvg(df):
        bull_fvg = (df['Low'] > df['High'].shift(2))
        bear_fvg = (df['High'] < df['Low'].shift(2))
        return bull_fvg, bear_fvg
    
    @staticmethod
    def check_ifvg_reversal(df):
        subset = df.iloc[-5:].copy().dropna()
        if len(subset) < 5: return None

        c1 = subset.iloc[0]
        c3 = subset.iloc[2]
        c5 = subset.iloc[4]
        
        is_bear_fvg_first = c3['High'] < c1['Low']
        is_bull_fvg_second = c5['Low'] > c3['High']
        
        if is_bear_fvg_first and is_bull_fvg_second:
            return 'Bull'

        is_bull_fvg_first = c3['Low'] > c1['High']
        is_bear_fvg_second = c5['High'] < c3['Low']
        
        if is_bull_fvg_first and is_bear_fvg_second:
            return 'Bear'
            
        return None

    @staticmethod
    def find_unmitigated_fvg_zone(df, threshold_pct=0.05):
        if len(df) < 5: return False
        current_price = df['Close'].iloc[-1]
        lookback = min(len(df), 50)
        
        for i in range(len(df)-1, len(df)-lookback, -1):
            curr_low = df['Low'].iloc[i]
            prev_high = df['High'].iloc[i-2]
            
            if curr_low > prev_high: 
                gap_top = curr_low
                gap_bottom = prev_high
                subsequent_data = df.iloc[i+1:]
                if not subsequent_data.empty:
                    if (subsequent_data['Low'] < gap_bottom).any():
                        continue 
                upper_bound = gap_bottom * (1 + threshold_pct)
                lower_bound = gap_bottom * (1 - threshold_pct)
                if lower_bound <= current_price <= upper_bound:
                    return True 
        return False

    @staticmethod
    def check_consecutive_candles(df, num_candles):
        if len(df) < num_candles:
            return None
        
        recent = df.iloc[-num_candles:]
        all_red = all(recent['Close'] < recent['Open'])
        all_green = all(recent['Close'] > recent['Open'])
        
        if all_red:
            return 'Bull'
        elif all_green:
            return 'Bear'
        
        return None

# ==========================================
# 5. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def fetch_bulk_data(tickers, interval="1d", period="2y"):
    if not tickers: return None
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=True, threads=True)
    return data

def resample_custom(df, timeframe):
    if df.empty: return df
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    try:
        if timeframe == "1D": return df.resample("1D").agg(agg_dict).dropna()
        if timeframe == "1W": return df.resample("W-FRI").agg(agg_dict).dropna()
        if timeframe == "1M": return df.resample("ME").agg(agg_dict).dropna()

        df_monthly = df.resample('MS').agg(agg_dict).dropna()
        if timeframe == "3M": return df_monthly.resample('QE').agg(agg_dict).dropna()
        if timeframe == "6M":
            df_monthly['Year'] = df_monthly.index.year
            df_monthly['Half'] = np.where(df_monthly.index.month <= 6, 1, 2)
            df_6m = df_monthly.groupby(['Year', 'Half']).agg(agg_dict)
            new_index = []
            for (year, half) in df_6m.index:
                month = 6 if half == 1 else 12
                new_index.append(pd.Timestamp(year=year, month=month, day=30))
            df_6m.index = pd.DatetimeIndex(new_index)
            return df_6m.sort_index()
        if timeframe == "12M": return df_monthly.resample('YE').agg(agg_dict).dropna()
    except: return df
    return df

# ==========================================
# 6. ANALYSIS ENGINE
# ==========================================

def count_all_signals(results, scan_type='bullish'):
    """Count total number of True signals across all scanners and timeframes."""
    count = 0
    
    if scan_type == 'bullish':
        signal_keys = [
            'OB_1D', 'OB_1W', 'OB_1M',
            'FVG_1D', 'FVG_1W', 'FVG_1M',
            'RevCand_1D', 'RevCand_1W', 'RevCand_1M',
            'iFVG_1D', 'iFVG_1W', 'iFVG_1M',
            'Support',
            'Squeeze_1D', 'Squeeze_1W'
        ]
    else:
        signal_keys = [
            'OB_1D', 'OB_1W', 'OB_1M',
            'FVG_1D', 'FVG_1W', 'FVG_1M',
            'RevCand_1D', 'RevCand_1W', 'RevCand_1M',
            'iFVG_1D', 'iFVG_1W', 'iFVG_1M',
            'Exhaustion'
        ]
    
    for key in signal_keys:
        if results.get(key, False):
            count += 1
    
    return count


def analyze_ticker(ticker, df_daily_raw, df_monthly_raw):
    """Runs ALL bullish scans for ONE ticker."""
    results = {
        'Ticker': ticker,
        'Price': 0,
        'OB_1D': False, 'OB_1W': False, 'OB_1M': False,
        'FVG_1D': False, 'FVG_1W': False, 'FVG_1M': False,
        'RevCand_1D': False, 'RevCand_1W': False, 'RevCand_1M': False,
        'iFVG_1D': False, 'iFVG_1W': False, 'iFVG_1M': False,
        'Support': False,
        'Squeeze_1D': False, 'Squeeze_1W': False,
        'has_signal': False,
        'signal_count': 0
    }
    
    try:
        d_1d = resample_custom(df_daily_raw, "1D")
        d_1w = resample_custom(df_daily_raw, "1W")
        d_1m = resample_custom(df_monthly_raw, "1M")
        d_3m = resample_custom(df_monthly_raw, "3M")
        d_6m = resample_custom(df_monthly_raw, "6M")
        d_12m = resample_custom(df_monthly_raw, "12M")
        
        data_map = {"1D": d_1d, "1W": d_1w, "1M": d_1m, "3M": d_3m, "6M": d_6m, "12M": d_12m}
        
        if not d_1d.empty:
            results['Price'] = round(d_1d['Close'].iloc[-1], 2)
        
    except: 
        return results

    for tf in ["1D", "1W", "1M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        
        df = data_map[tf].copy() 
        df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
        df['Is_Swing_High'], df['Is_Swing_Low'] = MathWiz.identify_strict_swings(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if curr['Bull_FVG']:
            past_swings = df[df['Is_Swing_High']]
            if not past_swings.empty:
                last_swing_high = past_swings['High'].iloc[-1]
                if curr['Close'] > last_swing_high and prev['Close'] <= last_swing_high:
                    results[f'FVG_{tf}'] = True
                    results['has_signal'] = True

        # Bullish Order Block: 1 bearish candle followed by 3 bullish candles + FVG on latest
        subset = df.iloc[-4:].copy()
        if len(subset) == 4:
            c1, c2, c3, c4 = subset.iloc[0], subset.iloc[1], subset.iloc[2], subset.iloc[3]
            # c1 must be bearish; c2, c3, c4 must be bullish
            c1_bearish = c1['Close'] < c1['Open']
            c2_bullish = c2['Close'] > c2['Open']
            c3_bullish = c3['Close'] > c3['Open']
            c4_bullish = c4['Close'] > c4['Open']
            # c4 must form a bullish FVG (c4 Low > c2 High)
            c4_has_bull_fvg = c4['Low'] > c2['High']
            
            if c1_bearish and c2_bullish and c3_bullish and c4_bullish and c4_has_bull_fvg:
                results[f'OB_{tf}'] = True
                results['has_signal'] = True
        
        ifvg_status = MathWiz.check_ifvg_reversal(df)
        if ifvg_status == "Bull":
            results[f'iFVG_{tf}'] = True
            results['has_signal'] = True

    sup_tf = []
    if MathWiz.find_unmitigated_fvg_zone(d_3m): sup_tf.append("3M")
    if MathWiz.find_unmitigated_fvg_zone(d_6m): sup_tf.append("6M")
    if MathWiz.find_unmitigated_fvg_zone(d_12m): sup_tf.append("12M")
    
    if len(sup_tf) >= 2:
        results['Support'] = True
        results['has_signal'] = True

    if not d_1d.empty:
        chop_series_d = MathWiz.calculate_choppiness(d_1d['High'], d_1d['Low'], d_1d['Close'])
        if not chop_series_d.empty and not pd.isna(chop_series_d.iloc[-1]):
            if chop_series_d.iloc[-1] > 59:
                results['Squeeze_1D'] = True
                results['has_signal'] = True
                
    if not d_1w.empty:
        chop_series_w = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series_w.empty and not pd.isna(chop_series_w.iloc[-1]):
            if chop_series_w.iloc[-1] > 59:
                results['Squeeze_1W'] = True
                results['has_signal'] = True

    if not d_1d.empty and len(d_1d) >= 5:
        if MathWiz.check_consecutive_candles(d_1d, 5) == 'Bull':
            results['RevCand_1D'] = True
            results['has_signal'] = True
    
    if not d_1w.empty and len(d_1w) >= 4:
        if MathWiz.check_consecutive_candles(d_1w, 4) == 'Bull':
            results['RevCand_1W'] = True
            results['has_signal'] = True
    
    if not d_1m.empty and len(d_1m) >= 3:
        if MathWiz.check_consecutive_candles(d_1m, 3) == 'Bull':
            results['RevCand_1M'] = True
            results['has_signal'] = True

    # Count all signals
    results['signal_count'] = count_all_signals(results, 'bullish')

    return results


def analyze_ticker_bearish(ticker, df_daily_raw, df_monthly_raw):
    """Runs ALL BEARISH scans for ONE ticker."""
    results = {
        'Ticker': ticker,
        'Price': 0,
        'OB_1D': False, 'OB_1W': False, 'OB_1M': False,
        'FVG_1D': False, 'FVG_1W': False, 'FVG_1M': False,
        'RevCand_1D': False, 'RevCand_1W': False, 'RevCand_1M': False,
        'iFVG_1D': False, 'iFVG_1W': False, 'iFVG_1M': False,
        'Exhaustion': False,
        'has_signal': False,
        'signal_count': 0
    }
    
    try:
        d_1d = resample_custom(df_daily_raw, "1D")
        d_1w = resample_custom(df_daily_raw, "1W")
        d_1m = resample_custom(df_monthly_raw, "1M")
        d_3m = resample_custom(df_monthly_raw, "3M")
        d_6m = resample_custom(df_monthly_raw, "6M")
        d_12m = resample_custom(df_monthly_raw, "12M")
        
        data_map = {"1D": d_1d, "1W": d_1w, "1M": d_1m, "3M": d_3m, "6M": d_6m, "12M": d_12m}
        
        if not d_1d.empty:
            results['Price'] = round(d_1d['Close'].iloc[-1], 2)
        
    except: 
        return results

    for tf in ["1D", "1W", "1M"]:
        if tf not in data_map or len(data_map[tf]) < 5: continue
        
        df = data_map[tf].copy() 
        df['Bull_FVG'], df['Bear_FVG'] = MathWiz.find_fvg(df)
        df['Is_Swing_High'], df['Is_Swing_Low'] = MathWiz.identify_strict_swings(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if curr['Bear_FVG']:
            past_swings = df[df['Is_Swing_Low']]
            if not past_swings.empty:
                last_swing_low = past_swings['Low'].iloc[-1]
                if curr['Close'] < last_swing_low and prev['Close'] >= last_swing_low:
                    results[f'FVG_{tf}'] = True
                    results['has_signal'] = True

        # Bearish Order Block: 1 bullish candle followed by 3 bearish candles + FVG on latest
        subset = df.iloc[-4:].copy()
        if len(subset) == 4:
            c1, c2, c3, c4 = subset.iloc[0], subset.iloc[1], subset.iloc[2], subset.iloc[3]
            # c1 must be bullish; c2, c3, c4 must be bearish
            c1_bullish = c1['Close'] > c1['Open']
            c2_bearish = c2['Close'] < c2['Open']
            c3_bearish = c3['Close'] < c3['Open']
            c4_bearish = c4['Close'] < c4['Open']
            # c4 must form a bearish FVG (c4 High < c2 Low)
            c4_has_bear_fvg = c4['High'] < c2['Low']
            
            if c1_bullish and c2_bearish and c3_bearish and c4_bearish and c4_has_bear_fvg:
                results[f'OB_{tf}'] = True
                results['has_signal'] = True
        
        ifvg_status = MathWiz.check_ifvg_reversal(df)
        if ifvg_status == "Bear":
            results[f'iFVG_{tf}'] = True
            results['has_signal'] = True

    if not d_1w.empty:
        chop_series = MathWiz.calculate_choppiness(d_1w['High'], d_1w['Low'], d_1w['Close'])
        if not chop_series.empty and not pd.isna(chop_series.iloc[-1]):
            if chop_series.iloc[-1] < 25:
                results['Exhaustion'] = True
                results['has_signal'] = True

    if not d_1d.empty and len(d_1d) >= 5:
        if MathWiz.check_consecutive_candles(d_1d, 5) == 'Bear':
            results['RevCand_1D'] = True
            results['has_signal'] = True
    
    if not d_1w.empty and len(d_1w) >= 4:
        if MathWiz.check_consecutive_candles(d_1w, 4) == 'Bear':
            results['RevCand_1W'] = True
            results['has_signal'] = True
    
    if not d_1m.empty and len(d_1m) >= 3:
        if MathWiz.check_consecutive_candles(d_1m, 3) == 'Bear':
            results['RevCand_1M'] = True
            results['has_signal'] = True

    # Count all signals
    results['signal_count'] = count_all_signals(results, 'bearish')

    return results


def create_consolidated_table(results_list, names_dict=None, scan_type='bullish'):
    """Creates a consolidated DataFrame with timeframes as main columns."""
    if not results_list:
        return None
    
    # Filter only stocks with at least one signal
    filtered = [r for r in results_list if r.get('has_signal', False)]
    
    if not filtered:
        return None
    
    # Sort by signal count descending by default
    filtered = sorted(filtered, key=lambda x: x.get('signal_count', 0), reverse=True)
    df = pd.DataFrame(filtered)
    
    if names_dict:
        df['Name'] = df['Ticker'].map(names_dict).fillna(df['Ticker'])
    
    check = "✅"
    empty = ""
    
    if scan_type == 'bullish':
        display_data = []
        for _, row in df.iterrows():
            display_row = {
                ('Info', 'Ticker'): row['Ticker'],
                ('Info', 'Name'): names_dict.get(row['Ticker'], row['Ticker']) if names_dict else row['Ticker'],
                ('Info', 'Price'): row['Price'],
                ('Info', 'Signals'): row.get('signal_count', 0),
                ('1D', 'OB'): check if row.get('OB_1D') else empty,
                ('1D', 'FVG'): check if row.get('FVG_1D') else empty,
                ('1D', 'Rev'): check if row.get('RevCand_1D') else empty,
                ('1D', 'iFVG'): check if row.get('iFVG_1D') else empty,
                ('1D', 'Sqz'): check if row.get('Squeeze_1D') else empty,
                ('1W', 'OB'): check if row.get('OB_1W') else empty,
                ('1W', 'FVG'): check if row.get('FVG_1W') else empty,
                ('1W', 'Rev'): check if row.get('RevCand_1W') else empty,
                ('1W', 'iFVG'): check if row.get('iFVG_1W') else empty,
                ('1W', 'Sqz'): check if row.get('Squeeze_1W') else empty,
                ('1M', 'OB'): check if row.get('OB_1M') else empty,
                ('1M', 'FVG'): check if row.get('FVG_1M') else empty,
                ('1M', 'Rev'): check if row.get('RevCand_1M') else empty,
                ('1M', 'iFVG'): check if row.get('iFVG_1M') else empty,
                ('LT', 'Sup'): check if row.get('Support') else empty,
            }
            display_data.append(display_row)
            
        display_df = pd.DataFrame(display_data)
        display_df.columns = pd.MultiIndex.from_tuples(display_df.columns)
            
    else:
        display_data = []
        for _, row in df.iterrows():
            display_row = {
                ('Info', 'Ticker'): row['Ticker'],
                ('Info', 'Name'): names_dict.get(row['Ticker'], row['Ticker']) if names_dict else row['Ticker'],
                ('Info', 'Price'): row['Price'],
                ('Info', 'Signals'): row.get('signal_count', 0),
                ('1D', 'OB'): check if row.get('OB_1D') else empty,
                ('1D', 'FVG'): check if row.get('FVG_1D') else empty,
                ('1D', 'Rev'): check if row.get('RevCand_1D') else empty,
                ('1D', 'iFVG'): check if row.get('iFVG_1D') else empty,
                ('1W', 'OB'): check if row.get('OB_1W') else empty,
                ('1W', 'FVG'): check if row.get('FVG_1W') else empty,
                ('1W', 'Rev'): check if row.get('RevCand_1W') else empty,
                ('1W', 'iFVG'): check if row.get('iFVG_1W') else empty,
                ('1W', 'Exh'): check if row.get('Exhaustion') else empty,
                ('1M', 'OB'): check if row.get('OB_1M') else empty,
                ('1M', 'FVG'): check if row.get('FVG_1M') else empty,
                ('1M', 'Rev'): check if row.get('RevCand_1M') else empty,
                ('1M', 'iFVG'): check if row.get('iFVG_1M') else empty,
            }
            display_data.append(display_row)
            
        display_df = pd.DataFrame(display_data)
        display_df.columns = pd.MultiIndex.from_tuples(display_df.columns)
    
    return display_df


# ==========================================
# 7. MAIN DASHBOARD UI
# ==========================================
def main():
    if not check_password():
        st.stop()
        
    st.title("🎯 Prath's Market Scanner")

    if 'init_done' not in st.session_state:
        st.cache_data.clear()
        st.session_state.init_done = True
    
    if 'scan_triggered' not in st.session_state:
        st.session_state.scan_triggered = False
    
    # Sidebar - Market Selection
    st.sidebar.header("🌍 Market Selection")
    
    country = st.sidebar.radio(
        "Select Country",
        ["🇺🇸 United States", "🇮🇳 India"],
        horizontal=True
    )
    
    is_india = "India" in country
    
    # Sector Selection based on country
    if is_india:
        sectors = INDIA_SECTORS
        st.sidebar.subheader("🏢 Select Sector")
    else:
        sectors = US_SECTORS
        st.sidebar.subheader("🏢 Select Sector")
    
    selected_sector = st.sidebar.selectbox(
        "Sector",
        list(sectors.keys()),
        index=0
    )
    
    # Get the filename for selected sector
    data_file = sectors[selected_sector]
    
    # Load tickers
    tickers, names_dict = load_tickers_from_file(data_file, is_india=is_india)
    
    # Scan Controls
    st.sidebar.divider()
    col_scan, col_refresh = st.sidebar.columns(2)
    with col_scan:
        if st.button("🎯 Run Scan", type="primary", use_container_width=True):
            st.session_state.scan_triggered = True
    with col_refresh:
        if st.button("🔄 Refresh", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.scan_triggered = False
            st.rerun()

    # Legend
    st.sidebar.divider()
    st.sidebar.subheader("📖 Legend")
    st.sidebar.markdown("""
    **OB** = Order Flow  
    **FVG** = Fair Value Gap  
    **Rev** = Reversal Candidate  
    **iFVG** = Inverse FVG  
    **Sqz** = Squeeze  
    **Sup** = Strong Support  
    **Exh** = Trend Exhaustion
    
    **Signals** = Total count of ✅
    """)

    # Main content
    market_name = "India (NSE)" if is_india else "United States"
    
    if not st.session_state.scan_triggered:
        st.info("👋 Welcome! Select a market and sector, then click **🎯 Run Scan** to analyze.")
        
        st.markdown(f"""
        ### 📊 Selected: {market_name} → {selected_sector}
        
        **Scan Types:** Order Flow, FVG, Reversal Candidates, iFVG, Strong Support, Squeeze
        
        **Timeframes:** 1D, 1W, 1M (Long-term for Support)
        
        **Output:** All stocks with at least 1 signal, sorted by signal count
        """)
        
        if tickers:
            st.success(f"✅ {len(tickers)} stocks loaded from **{selected_sector}**")
            
            # Show preview of tickers
            with st.expander("📋 View Ticker List"):
                preview_df = pd.DataFrame({
                    'Ticker': tickers[:50],
                    'Name': [names_dict.get(t, t) for t in tickers[:50]]
                })
                st.dataframe(preview_df, hide_index=True, use_container_width=True)
                if len(tickers) > 50:
                    st.caption(f"...and {len(tickers) - 50} more")
        else:
            st.warning("⚠️ No tickers found. Please check that the data files are present.")
        return

    # Run scan
    if tickers and st.session_state.scan_triggered:
        with st.status(f"🚀 Scanning {selected_sector}...", expanded=True) as status:
            
            st.write(f"Fetching data for {len(tickers)} tickers...")
            data_d = fetch_bulk_data(tickers, interval="1d", period="2y")
            data_m = fetch_bulk_data(tickers, interval="1mo", period="max")
            
            st.write("Processing scan algorithms...")
            
            bullish_results = []
            bearish_results = []
            is_multi = len(tickers) > 1
            
            with ThreadPoolExecutor() as executor:
                bull_futures = []
                bear_futures = []
                
                for ticker in tickers:
                    try:
                        df_d = data_d[ticker] if is_multi else data_d
                        df_m = data_m[ticker] if is_multi else data_m
                        
                        if not df_d.empty and not df_m.empty:
                            bull_futures.append(executor.submit(analyze_ticker, ticker, df_d, df_m))
                            bear_futures.append(executor.submit(analyze_ticker_bearish, ticker, df_d, df_m))
                    except: continue
                
                for f in bull_futures:
                    try:
                        res = f.result()
                        if res: bullish_results.append(res)
                    except: continue
                
                for f in bear_futures:
                    try:
                        res = f.result()
                        if res: bearish_results.append(res)
                    except: continue
            
            status.update(label="✅ Scan Complete", state="complete", expanded=False)

        # Results Header
        st.subheader(f"📊 Results: {market_name} → {selected_sector}")
        
        bull_table = create_consolidated_table(bullish_results, names_dict, 'bullish')
        bear_table = create_consolidated_table(bearish_results, names_dict, 'bearish')
        
        st.header("🐂 Bullish Scans")
        st.caption("Stocks with at least 1 bullish signal (sorted by signal count - click column header to re-sort)")
        
        if bull_table is not None and not bull_table.empty:
            st.dataframe(
                bull_table,
                hide_index=True,
                use_container_width=True,
                height=None  # No fixed height - show all rows
            )
        else:
            st.info("No bullish signals found.")
        
        st.divider()
        
        st.header("🐻 Bearish Scans")
        st.caption("Stocks with at least 1 bearish signal (sorted by signal count - click column header to re-sort)")
        
        if bear_table is not None and not bear_table.empty:
            st.dataframe(
                bear_table,
                hide_index=True,
                use_container_width=True,
                height=None  # No fixed height - show all rows
            )
        else:
            st.info("No bearish signals found.")
        
        # Summary
        st.divider()
        st.subheader("📊 Scan Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bull_count = len([r for r in bullish_results if r.get('has_signal')])
            st.metric("Bullish Stocks", bull_count)
        
        with col2:
            bear_count = len([r for r in bearish_results if r.get('has_signal')])
            st.metric("Bearish Stocks", bear_count)
        
        with col3:
            total_signals = sum(r.get('signal_count', 0) for r in bullish_results) + sum(r.get('signal_count', 0) for r in bearish_results)
            st.metric("Total Signals", total_signals)
        
        with col4:
            st.metric("Scanned", len(tickers))

if __name__ == "__main__":
    main()
