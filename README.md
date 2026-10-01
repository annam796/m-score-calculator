# m-score-calculator
M-score calculator with automated financial data retrieval using yfinance.

## Overview
This project fetches the latest two annual financial statements for a single stock ticker using `yfinance`, extracts the inputs required for the Beneish M-score, and computes the result using the provided manual M-score function.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the script:

```bash
python m_score_calculator.py AAPL
```

Or let the script prompt for a ticker:

```bash
python m_score_calculator.py
```

The script will:
- ask for a ticker symbol (or accept one as a command-line argument)
- retrieve the annual financial statements from Yahoo Finance
- extract all required M-score inputs
- compute the M-score using the provided `calculate_m_score_manual` function
- print the financial inputs used in the calculation

## Notes
- This implementation uses `yfinance` because it is simple to use for quick financial analysis scripts.
- Some companies may use slightly different statement labels than the common ones above. The lookup logic attempts to support several common variations.
