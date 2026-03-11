"""
MoneyMattersAI -- Expense Analytics Engine
============================================

This module analyzes categorized transactions, computes spending
statistics, and generates actionable financial insights.

It supports two modes:
  1. Pre-labeled data  -- CSV with 'transaction', 'amount', 'category'
  2. Unlabeled data    -- CSV with 'transaction', 'amount' only;
                          categories are predicted via the ML model.

Pipeline:
    transactions.csv
        -> load_transactions()
        -> classify (if needed) via predict_batch()
        -> analyze_spending()
        -> generate_spending_report()
        -> Structured JSON + printed report

Usage (CLI):
    python src/expense_analytics.py
    python src/expense_analytics.py --file data/transactions.csv
    python src/expense_analytics.py --json
    python src/expense_analytics.py --quiet --json

Usage (Import):
    from src.expense_analytics import analyze_spending, generate_spending_report
    report = generate_spending_report("data/transactions.csv")
"""

import os
import sys
import json
import argparse
import pandas as pd

# -- Resolve project root -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_batch


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data", "transactions.csv")

# Thresholds for insight generation
HIGH_SPEND_PCT = 30    # Alert if a category >= 30% of total
LOW_SPEND_PCT = 10     # Flag as "low" if a category <= 10%
SPIKE_MULTIPLIER = 2.5 # Flag transaction if > 2.5x category average


# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------
def _log(step: str, msg: str) -> None:
    print(f"  [{step}]  {msg}")


# =========================================================================
# CORE FUNCTION 1: load_transactions
# =========================================================================
def load_transactions(file_path: str = None) -> pd.DataFrame:
    """
    Load and validate transaction data from a CSV file.

    Expected columns:
        - transaction (required): Description of the expense.
        - amount      (required): Spending amount in INR.
        - category    (optional): If missing, ML model classifies them.

    Args:
        file_path: Path to CSV file. Defaults to data/transactions.csv.

    Returns:
        Validated DataFrame with columns [transaction, amount, category].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or data is empty.
    """
    if file_path is None:
        file_path = DEFAULT_DATASET

    # -- Check file exists -------------------------------------------------
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Transaction file not found: {file_path}\n"
            f"Please provide a valid CSV path."
        )

    # -- Read CSV ----------------------------------------------------------
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Transaction file is empty. No data to analyze.")

    # -- Validate required columns -----------------------------------------
    if "transaction" not in df.columns:
        raise ValueError(
            f"CSV must contain a 'transaction' column.\n"
            f"Found columns: {list(df.columns)}"
        )
    if "amount" not in df.columns:
        raise ValueError(
            f"CSV must contain an 'amount' column.\n"
            f"Found columns: {list(df.columns)}"
        )

    # -- Clean data --------------------------------------------------------
    # Ensure amount is numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # Drop rows with empty transaction descriptions
    df = df.dropna(subset=["transaction"])
    df = df[df["transaction"].str.strip() != ""]

    # -- Classify if no category column ------------------------------------
    if "category" not in df.columns or df["category"].isna().all():
        _log("ML", "No 'category' column found. Classifying via ML model...")
        predictions = predict_batch(df["transaction"].tolist(), verbose=False)
        df["category"] = [
            p.get("predicted_category", "Unknown") for p in predictions
        ]
        _log("ML", f"Classified {len(df)} transactions using the trained model.")

    # Drop rows where category is still missing
    df = df.dropna(subset=["category"])

    return df.reset_index(drop=True)


# =========================================================================
# CORE FUNCTION 2: analyze_spending
# =========================================================================
def analyze_spending(df: pd.DataFrame) -> dict:
    """
    Compute spending statistics from a categorized transaction DataFrame.

    Computes:
        - Total spending across all transactions
        - Total per category
        - Percentage distribution
        - Per-category stats (avg, min, max, count)
        - Top 5 most expensive transactions

    Args:
        df: DataFrame with columns [transaction, amount, category].

    Returns:
        Dictionary with keys:
            total_spending, total_transactions, category_breakdown,
            percentages, category_details, top_transactions
    """
    total_spending = float(df["amount"].sum())
    total_transactions = len(df)

    # -- Per-category totals -----------------------------------------------
    category_breakdown = {}
    percentages = {}
    category_details = {}

    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat]
        cat_total = float(cat_df["amount"].sum())
        cat_count = len(cat_df)
        cat_avg = float(cat_df["amount"].mean()) if cat_count > 0 else 0.0
        cat_min = float(cat_df["amount"].min()) if cat_count > 0 else 0.0
        cat_max = float(cat_df["amount"].max()) if cat_count > 0 else 0.0
        cat_pct = round(
            (cat_total / total_spending * 100) if total_spending > 0 else 0.0
        )

        category_breakdown[cat] = round(cat_total, 2)
        percentages[cat] = cat_pct

        category_details[cat] = {
            "total": round(cat_total, 2),
            "percentage": cat_pct,
            "transactions": cat_count,
            "average": round(cat_avg, 2),
            "min": round(cat_min, 2),
            "max": round(cat_max, 2),
        }

    # -- Top 5 transactions ------------------------------------------------
    top_transactions = (
        df.nlargest(5, "amount")[["transaction", "category", "amount"]]
        .to_dict(orient="records")
    )

    return {
        "total_spending": round(total_spending, 2),
        "total_transactions": total_transactions,
        "category_breakdown": category_breakdown,
        "percentages": percentages,
        "category_details": category_details,
        "top_transactions": top_transactions,
    }


# =========================================================================
# INSIGHT GENERATOR
# =========================================================================
def _generate_insights(analytics: dict, df: pd.DataFrame) -> list:
    """
    Produce actionable financial insights from spending analytics.

    Generates 7 types of insights:
        1. Total spending summary
        2. Highest spending category
        3. Lowest spending category
        4. High-spending alerts (> 30% threshold)
        5. Low-spending observations (< 10%)
        6. Spending spike detection (single txns > 2.5x avg)
        7. Category comparison

    Args:
        analytics: Output dict from analyze_spending().
        df: Original DataFrame (for spike detection).

    Returns:
        List of insight strings.
    """
    insights = []
    total = analytics["total_spending"]
    details = analytics["category_details"]
    pcts = analytics["percentages"]

    if total == 0:
        return ["No spending data available for analysis."]

    # -- 1. Summary --------------------------------------------------------
    insights.append(
        f"Total spending: Rs.{total:,.2f} across "
        f"{analytics['total_transactions']} transactions."
    )

    # -- 2. Highest category -----------------------------------------------
    highest = max(details, key=lambda c: details[c]["total"])
    insights.append(
        f"{highest} is your highest spending category "
        f"at {pcts[highest]}% (Rs.{details[highest]['total']:,.2f})."
    )

    # -- 3. Lowest category ------------------------------------------------
    lowest = min(details, key=lambda c: details[c]["total"])
    insights.append(
        f"{lowest} is your lowest spending category "
        f"at {pcts[lowest]}% (Rs.{details[lowest]['total']:,.2f})."
    )

    # -- 4. High-spending alerts -------------------------------------------
    for cat, stats in details.items():
        if stats["percentage"] >= HIGH_SPEND_PCT:
            insights.append(
                f"[ALERT] You spent {stats['percentage']}% of your budget on "
                f"{cat}. Consider reviewing these expenses."
            )

    # -- 5. Low-spending categories ----------------------------------------
    for cat, stats in details.items():
        if stats["percentage"] <= LOW_SPEND_PCT:
            insights.append(
                f"{cat} expenses are relatively low ({stats['percentage']}%). "
                f"This category is well-managed."
            )

    # -- 6. Spike detection ------------------------------------------------
    for cat, stats in details.items():
        avg = stats["average"]
        if avg == 0:
            continue
        spikes = df[
            (df["category"] == cat) &
            (df["amount"] > avg * SPIKE_MULTIPLIER)
        ]
        for _, row in spikes.iterrows():
            insights.append(
                f"[SPIKE] \"{row['transaction']}\" (Rs.{row['amount']:,.2f}) "
                f"is {row['amount'] / avg:.1f}x your {cat} average of "
                f"Rs.{avg:,.2f}."
            )

    # -- 7. Top vs second category comparison ------------------------------
    sorted_cats = sorted(
        details.keys(), key=lambda c: details[c]["total"], reverse=True
    )
    if len(sorted_cats) >= 2:
        first, second = sorted_cats[0], sorted_cats[1]
        diff = details[first]["total"] - details[second]["total"]
        insights.append(
            f"You spend Rs.{diff:,.2f} more on {first} than on {second}."
        )

    return insights


# =========================================================================
# CORE FUNCTION 3: generate_spending_report
# =========================================================================
def generate_spending_report(file_path: str = None, verbose: bool = True) -> dict:
    """
    Run the full expense analytics pipeline end-to-end.

    Steps:
        1. Load transactions from CSV
        2. Classify via ML model (if categories are missing)
        3. Compute spending analytics
        4. Generate financial insights

    Args:
        file_path: Path to CSV file. Defaults to data/transactions.csv.
        verbose:   If True, print progress logs.

    Returns:
        Complete analytics report as a structured dictionary:
        {
            "total_spending": 7850,
            "category_breakdown": {"Food": 2100, ...},
            "percentages": {"Food": 26, ...},
            "insights": ["Shopping is your highest...", ...]
        }
    """
    if verbose:
        print()
        print("=" * 60)
        print("  MoneyMattersAI -- Expense Analytics Report")
        print("=" * 60)
        print()

    # -- 1. Load -----------------------------------------------------------
    if verbose:
        _log("1/4", "Loading transaction data...")
    df = load_transactions(file_path)
    if verbose:
        _log("1/4", f"Loaded {len(df)} transactions.")
        _log("1/4", f"Categories found: {', '.join(sorted(df['category'].unique()))}")

    # -- 2. Verify classification ------------------------------------------
    if verbose:
        _log("2/4", "Verifying transaction categories...")
    # Run ML predictions for comparison / validation
    predictions = predict_batch(df["transaction"].tolist(), verbose=False)
    matches = sum(
        1 for pred, actual in zip(predictions, df["category"])
        if pred.get("predicted_category") == actual
    )
    if verbose:
        _log("2/4", f"ML model agrees with {matches}/{len(df)} "
                     f"({matches/len(df):.0%}) category labels.")

    # -- 3. Analyze --------------------------------------------------------
    if verbose:
        _log("3/4", "Computing spending analytics...")
    analytics = analyze_spending(df)

    # -- 4. Insights -------------------------------------------------------
    if verbose:
        _log("4/4", "Generating financial insights...")
    insights = _generate_insights(analytics, df)
    analytics["insights"] = insights

    if verbose:
        _log("DONE", "Analytics report generated successfully.")

    return analytics


# =========================================================================
# IN-MEMORY ANALYSIS (used by API endpoint /analyze)
# =========================================================================
def analyze_from_records(records: list) -> dict:
    """
    Run the full analytics pipeline directly on a list of dicts.

    This bypasses file I/O entirely — perfect for REST API endpoints
    that receive JSON payloads instead of CSV uploads.

    Each record must have:
        - "transaction" (str):  Description of the expense.
        - "amount"      (float): Spending amount in INR.
        - "category"    (str, optional): If omitted, ML model predicts it.

    Args:
        records: List of transaction dicts, e.g.:
            [
                {"transaction": "zomato dinner", "amount": 350},
                {"transaction": "uber ride",     "amount": 220},
                {"transaction": "amazon order",  "amount": 1200, "category": "Shopping"},
            ]

    Returns:
        Full analytics report dict (same structure as generate_spending_report()).

    Raises:
        ValueError: If records list is empty or missing required fields.
    """
    if not records:
        raise ValueError("Transaction list is empty. Provide at least one record.")

    # -- Validate and build DataFrame --------------------------------------
    for i, rec in enumerate(records):
        if "transaction" not in rec or not str(rec["transaction"]).strip():
            raise ValueError(
                f"Record at index {i} is missing a valid 'transaction' field."
            )
        if "amount" not in rec:
            raise ValueError(
                f"Record at index {i} ('{rec['transaction']}') is missing 'amount'."
            )
        try:
            float(rec["amount"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Record at index {i} has an invalid 'amount': {rec['amount']!r}. "
                f"Amount must be a numeric value."
            )

    df = pd.DataFrame(records)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # -- Auto-classify if no category --------------------------------------
    if "category" not in df.columns or df["category"].isna().all():
        predictions = predict_batch(df["transaction"].tolist(), verbose=False)
        df["category"] = [
            p.get("predicted_category", "Unknown") for p in predictions
        ]
    else:
        # Partially fill missing categories
        missing_mask = df["category"].isna() | (df["category"].str.strip() == "")
        if missing_mask.any():
            missing_txns = df.loc[missing_mask, "transaction"].tolist()
            preds = predict_batch(missing_txns, verbose=False)
            df.loc[missing_mask, "category"] = [
                p.get("predicted_category", "Unknown") for p in preds
            ]

    # -- Run analytics pipeline --------------------------------------------
    analytics = analyze_spending(df)
    analytics["insights"] = _generate_insights(analytics, df)

    return analytics


# =========================================================================
# PRETTY PRINTER
# =========================================================================
def print_report(report: dict) -> None:
    """Display the analytics report in a formatted, readable layout."""

    print()
    print("-" * 60)
    print("  SPENDING SUMMARY")
    print("-" * 60)
    print(f"  Total Spending:      Rs.{report['total_spending']:>12,.2f}")
    print(f"  Total Transactions:  {report['total_transactions']:>12}")
    print()

    # -- Category Breakdown ------------------------------------------------
    print("-" * 60)
    print("  CATEGORY BREAKDOWN")
    print("-" * 60)
    print(f"  {'Category':<14} {'Total (Rs.)':>12} {'Pct':>6} "
          f"{'Txns':>6} {'Avg (Rs.)':>12}")
    print("  " + "-" * 54)

    details = report["category_details"]
    for cat in sorted(details, key=lambda c: details[c]["total"], reverse=True):
        s = details[cat]
        bar = "#" * max(1, int(s["percentage"] / 100 * 30))
        print(
            f"  {cat:<14} {s['total']:>12,.2f} {s['percentage']:>5}% "
            f"{s['transactions']:>6} {s['average']:>12,.2f}"
        )
        print(f"  {'':14} {bar}")

    print()

    # -- Percentage Distribution -------------------------------------------
    print("-" * 60)
    print("  PERCENTAGE DISTRIBUTION")
    print("-" * 60)
    for cat in sorted(report["percentages"],
                      key=lambda c: report["percentages"][c], reverse=True):
        pct = report["percentages"][cat]
        bar = "=" * max(1, int(pct / 100 * 40))
        print(f"  {cat:<14} {pct:>3}%  |{bar}|")
    print()

    # -- Top Transactions --------------------------------------------------
    print("-" * 60)
    print("  TOP 5 TRANSACTIONS")
    print("-" * 60)
    for i, txn in enumerate(report["top_transactions"], 1):
        print(
            f"  {i}. {txn['transaction']:<30} "
            f"{txn['category']:<12} "
            f"Rs.{txn['amount']:>10,.2f}"
        )
    print()

    # -- Insights ----------------------------------------------------------
    print("-" * 60)
    print("  SPENDING INSIGHTS")
    print("-" * 60)
    for i, insight in enumerate(report["insights"], 1):
        print(f"  {i:>2}. {insight}")

    print()
    print("=" * 60)
    print()


# =========================================================================
# CLI ENTRY POINT
# =========================================================================
def main():
    """
    Command-line interface for expense analytics.

    Usage:
        python src/expense_analytics.py
        python src/expense_analytics.py --file data/transactions.csv
        python src/expense_analytics.py --json
        python src/expense_analytics.py --quiet --json
    """
    parser = argparse.ArgumentParser(
        prog="MoneyMattersAI Analytics",
        description=(
            "Analyze categorized transactions and generate spending insights."
        ),
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to transactions CSV (default: data/transactions.csv).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the full report as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step logs.",
    )

    args = parser.parse_args()

    try:
        report = generate_spending_report(
            file_path=args.file,
            verbose=not args.quiet,
        )

        if args.json:
            # Clean JSON output (remove category_details for compact view)
            json_output = {
                "total_spending": report["total_spending"],
                "category_breakdown": report["category_breakdown"],
                "percentages": report["percentages"],
                "insights": report["insights"],
            }
            print(json.dumps(json_output, indent=2))
        else:
            print_report(report)

    except FileNotFoundError as e:
        print(f"\n  [ERROR] FILE NOT FOUND: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n  [ERROR] DATA ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] UNEXPECTED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
