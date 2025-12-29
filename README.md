# 🎯 Prath's Market Scanner

A comprehensive multi-timeframe market scanner for US and Indian markets, built with Streamlit and yfinance.

## Features

- **Multi-Market Support**: Scan US (NYSE/NASDAQ) and Indian (NSE) markets
- **Sector-Based Scanning**: Choose from 11+ sectors in each market
- **Multiple Scan Types**:
  - Order Flow (Bullish/Bearish Order Blocks)
  - Fair Value Gap (FVG) Breakouts/Breakdowns
  - Reversal Candidates (Consecutive candle patterns)
  - Inverse FVG (iFVG) Reversals
  - Strong Support Zones
  - Volatility Squeeze
  - Trend Exhaustion
- **Multi-Timeframe Analysis**: 1D, 1W, 1M, and Long-term
- **Cross-Scanner Confluence**: Only shows stocks with signals from 2+ different scan types

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/market-scanner.git
cd market-scanner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Configuration

### Password Protection
The app is password-protected. Default password: `Sniper2025`

To change the password:
1. Create `.streamlit/secrets.toml` in your repo root
2. Add: `PASSWORD = "your_new_password"`

### Deployment on Streamlit Cloud
1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file path as `app.py`
5. Add your `PASSWORD` secret in the Streamlit Cloud dashboard

## File Structure

```
market-scanner/
├── app.py                          # Main application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── secrets.toml               # Password configuration
│
├── # US Market Data Files
├── us_sectors_indices.csv         # US Sector ETFs & Major Indices
├── us_technology.csv              # Technology sector stocks
├── us_healthcare.csv              # Healthcare sector stocks
├── us_financials.csv              # Financials sector stocks
├── us_consumer_discretionary.csv  # Consumer Discretionary stocks
├── us_consumer_staples.csv        # Consumer Staples stocks
├── us_energy.csv                  # Energy sector stocks
├── us_industrials.csv             # Industrials sector stocks
├── us_materials.csv               # Materials sector stocks
├── us_utilities.csv               # Utilities sector stocks
├── us_real_estate.csv             # Real Estate sector stocks
├── us_communication.csv           # Communication Services stocks
│
├── # India Market Data Files
├── ind_sectors_indices.csv        # India Sector Indices & ETFs
├── ind_nifty50.csv                # Nifty 50 constituents
├── ind_nifty_bank.csv             # Nifty Bank constituents
├── ind_nifty_it.csv               # Nifty IT constituents
├── ind_nifty_pharma.csv           # Nifty Pharma constituents
├── ind_nifty_auto.csv             # Nifty Auto constituents
├── ind_nifty_fmcg.csv             # Nifty FMCG constituents
├── ind_nifty_metal.csv            # Nifty Metal constituents
├── ind_nifty_realty.csv           # Nifty Realty constituents
├── ind_nifty_energy.csv           # Nifty Energy constituents
├── ind_nifty_infra.csv            # Nifty Infrastructure constituents
├── ind_nifty_psu_bank.csv         # Nifty PSU Bank constituents
├── ind_nifty_private_bank.csv     # Nifty Private Bank constituents
├── ind_nifty_media.csv            # Nifty Media constituents
├── ind_nifty_midcap100.csv        # Nifty Midcap 100 constituents
└── ind_nifty_smallcap100.csv      # Nifty Smallcap 100 constituents
```

## Scan Legend

| Abbreviation | Full Name | Description |
|--------------|-----------|-------------|
| OB | Order Flow | Order Block patterns (3 consecutive same-color candles after opposite anchor) |
| FVG | Fair Value Gap | Price gaps with swing break confirmation |
| Rev | Reversal Candidate | 5+ red (1D), 4+ red (1W), or 3+ red (1M) consecutive candles |
| iFVG | Inverse FVG | 5-candle V-shape pattern (FVG drop followed by FVG pop, or vice versa) |
| Sqz | Squeeze | Choppiness Index > 59 (volatility compression) |
| Sup | Strong Support | Unmitigated FVG zones on 2+ long-term timeframes (3M/6M/12M) |
| Exh | Trend Exhaustion | Choppiness Index < 25 on weekly (trend exhaustion) |

## Usage

1. **Select Country**: Choose between US or India
2. **Select Sector**: Pick a specific sector or the overview (indices/ETFs)
3. **Run Scan**: Click "🎯 Run Scan" to analyze
4. **View Results**: 
   - Bullish table shows stocks with bullish confluence
   - Bearish table shows stocks with bearish confluence
   - Only stocks with 2+ different scan types triggering are displayed

## Notes

- Data is fetched from Yahoo Finance via yfinance
- Results are cached for 1 hour to improve performance
- Indian stocks automatically get `.NS` suffix for NSE
- The scanner uses parallel processing for faster analysis

## License

MIT License - Feel free to use and modify.

## Disclaimer

This tool is for educational and informational purposes only. It is not financial advice. Always do your own research before making investment decisions.
