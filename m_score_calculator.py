import sys
from typing import Iterable, Optional, Tuple, Dict, Any

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------
# Exact M-score function provided by the user
# ---------------------------------------------------------------------
def calculate_m_score_manual(
    receivables_current,
    revenue_current,
    gross_profit_current,
    current_assets_current,
    total_assets_current,
    ppe_current,
    depreciation_current,
    sga_expenses_current,
    current_liabilities_current,
    long_term_debt_current,
    net_income_current,
    nonoperating_income_current,
    operating_cash_flow_current,
    receivables_previous,
    revenue_previous,
    gross_profit_previous,
    current_assets_previous,
    total_assets_previous,
    ppe_previous,
    depreciation_previous,
    sga_expenses_previous,
    current_liabilities_previous,
    long_term_debt_previous,
    net_income_previous,
    nonoperating_income_previous,
    operating_cash_flow_previous
):
    # DSRI: Days Sales in Receivables Index
    dsri = (receivables_current / revenue_current) / (receivables_previous / revenue_previous)

    # GMI: Gross Margin Index
    gmi = (gross_profit_previous / revenue_previous) / (gross_profit_current / revenue_current)

    # AQI: Asset Quality Index
    aqi = (
        (1 - (current_assets_current + ppe_current) / total_assets_current)
        / (1 - (current_assets_previous + ppe_previous) / total_assets_previous)
    )

    # SGI: Sales Growth Index
    sgi = revenue_current / revenue_previous

    # DEPI: Depreciation Index
    depi = (
        (depreciation_previous / (ppe_previous + depreciation_previous))
        / (depreciation_current / (ppe_current + depreciation_current))
    )

    # SGAI: Sales, General, and Administrative Expenses Index
    sgai = (sga_expenses_current / revenue_current) / (sga_expenses_previous / revenue_previous)

    # LVGI: Leverage Index
    lvgi = (
        ((current_liabilities_current + long_term_debt_current) / total_assets_current)
        / ((current_liabilities_previous + long_term_debt_previous) / total_assets_previous)
    )

    # TATA: Total Accruals to Total Assets
    if operating_cash_flow_current is not None:
        tata = (net_income_current - nonoperating_income_current - operating_cash_flow_current) / total_assets_current
    else:
        tata = None

    # Calculate the M-Score
    if tata is not None:
        m_score = (
            -4.84
            + 0.92 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * tata
            - 0.327 * lvgi
        )
    else:
        m_score = None

    return m_score


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def clean_numeric(value: Any) -> Optional[float]:
    if value is None or value is pd.NA:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").strip()
        if cleaned in ("", "-", "nan", "NaN"):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_label(label: str) -> str:
    return "".join(ch.lower() for ch in str(label) if ch.isalnum())


def find_matching_label(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None

    # Exact normalized match
    exact_map = {normalize_label(str(label)): str(label) for label in df.index}
    for candidate in candidates:
        key = normalize_label(candidate)
        if key in exact_map:
            return exact_map[key]

    # Partial match fallback
    labels = [str(label) for label in df.index]
    for candidate in candidates:
        c_norm = normalize_label(candidate)
        for label in labels:
            l_norm = normalize_label(label)
            if c_norm in l_norm or l_norm in c_norm:
                return label

    return None


def get_value_for_year(df: pd.DataFrame, candidates: Iterable[str], year: Any) -> Optional[float]:
    if df is None or df.empty:
        return None

    label = find_matching_label(df, candidates)
    if label is None:
        return None

    try:
        value = df.loc[label, year]
    except KeyError:
        return None

    return clean_numeric(value)


def sort_years_desc(columns) -> list:
    def year_key(value):
        s = str(value)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else -1

    return sorted(columns, key=year_key, reverse=True)


def debug_print_labels(ticker_symbol: str):
    """Print all available labels in the financial statements for debugging."""
    ticker = yf.Ticker(ticker_symbol)
    
    print(f"\n--- DEBUGGING: Available labels for {ticker_symbol} ---\n")
    
    if ticker.income_stmt is not None and not ticker.income_stmt.empty:
        print("Income Statement labels:")
        for label in ticker.income_stmt.index:
            print(f"  - {label}")
    
    if ticker.balance_sheet is not None and not ticker.balance_sheet.empty:
        print("\nBalance Sheet labels:")
        for label in ticker.balance_sheet.index:
            print(f"  - {label}")
    
    if ticker.cashflow is not None and not ticker.cashflow.empty:
        print("\nCash Flow labels:")
        for label in ticker.cashflow.index:
            print(f"  - {label}")
    
    print("\n--- END DEBUG ---\n")


def fetch_m_score_inputs(ticker_symbol: str) -> Dict[str, Any]:
    ticker = yf.Ticker(ticker_symbol)

    try:
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
    except Exception as exc:
        raise ValueError(f"Unable to load financial statements for {ticker_symbol}: {exc}") from exc

    if income_stmt is None or balance_sheet is None or cashflow is None:
        raise ValueError(f"Missing one or more financial statement tables for {ticker_symbol}.")

    years = sort_years_desc(balance_sheet.columns)
    if len(years) < 2:
        raise ValueError(f"Need at least two years of data for {ticker_symbol}.")

    current_year = years[0]
    previous_year = years[1]

    def balance_value(candidates: Iterable[str], use_current: bool = True):
        year = current_year if use_current else previous_year
        return get_value_for_year(balance_sheet, candidates, year)

    def income_value(candidates: Iterable[str], use_current: bool = True):
        year = current_year if use_current else previous_year
        return get_value_for_year(income_stmt, candidates, year)

    def cashflow_value(candidates: Iterable[str], use_current: bool = True):
        year = current_year if use_current else previous_year
        return get_value_for_year(cashflow, candidates, year)

    inputs = {
        "current_year": current_year,
        "previous_year": previous_year,

        # Current year
        "receivables_current": balance_value(["Accounts Receivable", "Net Receivables", "Total Receivables"]),
        "revenue_current": income_value(["Total Revenue", "Revenue", "Operating Revenue"]),
        "gross_profit_current": income_value(["Gross Profit"]),
        "current_assets_current": balance_value(["Current Assets"]),
        "total_assets_current": balance_value(["Total Assets"]),
        "ppe_current": balance_value([
            "Property Plant Equipment Net",
            "Net PP&E",
            "Property Plant and Equipment Net",
            "Net Property Plant Equipment",
            "PP&E Net",
            "Fixed Assets",
            "Net Property Plant And Equipment",
        ]),
        "depreciation_current": cashflow_value([
            "Depreciation",
            "Depreciation and Amortization",
            "Depreciation Amortization Depletion",
        ]),
        "sga_expenses_current": income_value([
            "Selling General and Administrative",
            "Selling, General and Administrative",
            "Selling General Administrative Expense",
            "General and Administrative Expense",
            "Selling and Administrative Expense",
            "SG&A Expense",
            "Operating Expenses",
        ]),
        "current_liabilities_current": balance_value(["Current Liabilities"]),
        "long_term_debt_current": balance_value([
            "Long Term Debt",
            "Long-Term Debt",
            "Long Term Debt and Current Portion of Long Term Debt",
            "Noncurrent Liabilities",
            "Total Debt",
        ]),
        "net_income_current": income_value(["Net Income", "Net Income Common Stockholders"]),
        "nonoperating_income_current": income_value([
            "Non Operating Income Expense",
            "Non-Operating Income",
            "Other Income Expense",
            "Other Income",
            "Non Operating Income",
        ]),
        "operating_cash_flow_current": cashflow_value([
            "Operating Cash Flow",
            "Net Cash Provided by Operating Activities",
            "Net Cash From Operating Activities",
        ]),

        # Previous year
        "receivables_previous": balance_value(["Accounts Receivable", "Net Receivables", "Total Receivables"], use_current=False),
        "revenue_previous": income_value(["Total Revenue", "Revenue", "Operating Revenue"], use_current=False),
        "gross_profit_previous": income_value(["Gross Profit"], use_current=False),
        "current_assets_previous": balance_value(["Current Assets"], use_current=False),
        "total_assets_previous": balance_value(["Total Assets"], use_current=False),
        "ppe_previous": balance_value([
            "Property Plant Equipment Net",
            "Net PP&E",
            "Property Plant and Equipment Net",
            "Net Property Plant Equipment",
            "PP&E Net",
            "Fixed Assets",
            "Net Property Plant And Equipment",
        ], use_current=False),
        "depreciation_previous": cashflow_value([
            "Depreciation",
            "Depreciation and Amortization",
            "Depreciation Amortization Depletion",
        ], use_current=False),
        "sga_expenses_previous": income_value([
            "Selling General and Administrative",
            "Selling, General and Administrative",
            "Selling General Administrative Expense",
            "General and Administrative Expense",
            "Selling and Administrative Expense",
            "SG&A Expense",
            "Operating Expenses",
        ], use_current=False),
        "current_liabilities_previous": balance_value(["Current Liabilities"], use_current=False),
        "long_term_debt_previous": balance_value([
            "Long Term Debt",
            "Long-Term Debt",
            "Long Term Debt and Current Portion of Long Term Debt",
            "Noncurrent Liabilities",
            "Total Debt",
        ], use_current=False),
        "net_income_previous": income_value(["Net Income", "Net Income Common Stockholders"], use_current=False),
        "nonoperating_income_previous": income_value([
            "Non Operating Income Expense",
            "Non-Operating Income",
            "Other Income Expense",
            "Other Income",
            "Non Operating Income",
        ], use_current=False),
        "operating_cash_flow_previous": cashflow_value([
            "Operating Cash Flow",
            "Net Cash Provided by Operating Activities",
            "Net Cash From Operating Activities",
        ], use_current=False),
    }

    return inputs


def calculate_m_score_for_ticker(ticker_symbol: str) -> Tuple[Optional[float], Dict[str, Any]]:
    inputs = fetch_m_score_inputs(ticker_symbol)

    required_keys = [
        "receivables_current",
        "revenue_current",
        "gross_profit_current",
        "current_assets_current",
        "total_assets_current",
        "ppe_current",
        "depreciation_current",
        "sga_expenses_current",
        "current_liabilities_current",
        "long_term_debt_current",
        "net_income_current",
        "nonoperating_income_current",
        "operating_cash_flow_current",
        "receivables_previous",
        "revenue_previous",
        "gross_profit_previous",
        "current_assets_previous",
        "total_assets_previous",
        "ppe_previous",
        "depreciation_previous",
        "sga_expenses_previous",
        "current_liabilities_previous",
        "long_term_debt_previous",
        "net_income_previous",
        "nonoperating_income_previous",
        "operating_cash_flow_previous",
    ]

    missing = [key for key in required_keys if inputs.get(key) is None]
    if missing:
        print(f"\nDEBUG: Missing keys: {missing}")
        debug_print_labels(ticker_symbol)
        raise ValueError(
            f"Missing required inputs for {ticker_symbol}: {', '.join(missing)}. "
            "The data may not match the expected labels for this company or the statement may not include all values."
        )

    m_score = calculate_m_score_manual(
        inputs["receivables_current"],
        inputs["revenue_current"],
        inputs["gross_profit_current"],
        inputs["current_assets_current"],
        inputs["total_assets_current"],
        inputs["ppe_current"],
        inputs["depreciation_current"],
        inputs["sga_expenses_current"],
        inputs["current_liabilities_current"],
        inputs["long_term_debt_current"],
        inputs["net_income_current"],
        inputs["nonoperating_income_current"],
        inputs["operating_cash_flow_current"],
        inputs["receivables_previous"],
        inputs["revenue_previous"],
        inputs["gross_profit_previous"],
        inputs["current_assets_previous"],
        inputs["total_assets_previous"],
        inputs["ppe_previous"],
        inputs["depreciation_previous"],
        inputs["sga_expenses_previous"],
        inputs["current_liabilities_previous"],
        inputs["long_term_debt_previous"],
        inputs["net_income_previous"],
        inputs["nonoperating_income_previous"],
        inputs["operating_cash_flow_previous"]
    )

    return m_score, inputs


def main():
    if len(sys.argv) > 1:
        ticker = sys.argv[1].strip().upper()
    else:
        ticker = input("Enter stock ticker (for example: AAPL): ").strip().upper()

    if not ticker:
        print("No ticker entered.")
        return

    print(f"Fetching financial data for {ticker}...")
    try:
        m_score, inputs = calculate_m_score_for_ticker(ticker)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    print("\nFinancial inputs used in the M-score calculation:")
    for key in [
        "current_year",
        "previous_year",
        "receivables_current",
        "revenue_current",
        "gross_profit_current",
        "current_assets_current",
        "total_assets_current",
        "ppe_current",
        "depreciation_current",
        "sga_expenses_current",
        "current_liabilities_current",
        "long_term_debt_current",
        "net_income_current",
        "nonoperating_income_current",
        "operating_cash_flow_current",
        "receivables_previous",
        "revenue_previous",
        "gross_profit_previous",
        "current_assets_previous",
        "total_assets_previous",
        "ppe_previous",
        "depreciation_previous",
        "sga_expenses_previous",
        "current_liabilities_previous",
        "long_term_debt_previous",
        "net_income_previous",
        "nonoperating_income_previous",
        "operating_cash_flow_previous",
    ]:
        value = inputs.get(key)
        if value is not None:
            print(f"  {key}: {value}")

    print(f"\nM-Score for {ticker}: {m_score}")


if __name__ == "__main__":
    main()
