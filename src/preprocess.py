"""
MoneyMattersAI — Preprocessing Module
======================================

This module provides text cleaning utilities for transaction descriptions.
It is used by both the training pipeline (train_model.py) and the
prediction pipeline (predict.py) to ensure consistent text normalization.

Preprocessing Steps:
    1. Convert to lowercase
    2. Remove numbers
    3. Remove punctuation
    4. Collapse extra whitespace and strip edges
"""

import re
import string


def clean_transaction_text(text: str) -> str:
    """
    Clean and normalize a transaction description for ML processing.

    This function applies a series of text transformations to standardize
    transaction strings before they are vectorized by TF-IDF. Using the
    same cleaning logic in both training and prediction ensures that the
    model sees consistent input at every stage.

    Args:
        text (str): Raw transaction description.
                    Example: "Zomato Order #123"

    Returns:
        str: Cleaned transaction text.
             Example: "zomato order"

    Raises:
        ValueError: If text is None or not a string.

    Examples:
        >>> clean_transaction_text("Zomato Order #123")
        'zomato order'
        >>> clean_transaction_text("UBER ride — ₹250")
        'uber ride'
        >>> clean_transaction_text("  Amazon   Purchase!!  ")
        'amazon purchase'
    """

    # ── Guard clause: validate input ────────────────────────────────────
    if text is None:
     raise ValueError("Input text cannot be None.")

    if not isinstance(text, str):
     raise ValueError(f"Expected a string, got {type(text).__name__}.")

    text = text.strip()

    # ── Step 1: Convert to lowercase ────────────────────────────────────
    # Ensures "Zomato" and "zomato" are treated identically by the model.
    text = text.lower()

    # ── Step 2: Remove numbers ──────────────────────────────────────────
    # Transaction IDs, amounts, and order numbers add noise, not meaning.
    # Example: "order #123" → "order #"
    text = re.sub(r"\d+", "", text)

    # ── Step 3: Remove punctuation ──────────────────────────────────────
    # Symbols like #, !, ₹, — carry no semantic value for classification.
    # We remove all standard ASCII punctuation plus common Unicode symbols.
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"[^\w\s]", "", text)  # Catch remaining Unicode symbols (₹, —, etc.)

    # ── Step 4: Collapse extra whitespace ───────────────────────────────
    # After removals, multiple spaces may remain. Normalize to single space.
    # Example: "uber   ride" → "uber ride"
    text = re.sub(r"\s+", " ", text).strip()

    # ── Step 5: Remove duplicate words (important for merged datasets) ──
    text = " ".join(dict.fromkeys(text.split()))

    if text == "":
        return "unknown"

    return text


# ─────────────────────────────────────────────────────────────────────────
# Quick sanity check when running the module directly
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("Zomato Order #123", "zomato order"),
        ("UBER ride — ₹250", "uber ride"),
        ("  Amazon   Purchase!!  ", "amazon purchase"),
        ("Electricity Bill Payment 2024", "electricity bill payment"),
        ("flipkart ORDER #FLP-98765", "flipkart order flp"),
        ("", "unknown"),
    ]

    print("=" * 55)
    print("  MoneyMattersAI — Preprocessing Sanity Check")
    print("=" * 55)

    all_passed = True
    for raw, expected in test_cases:
        result = clean_transaction_text(raw)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status}  \"{raw}\"")
        print(f"       → \"{result}\"")
        if result != expected:
            print(f"       ✗ expected \"{expected}\"")
        print()

    print("=" * 55)
    print(f"  Result: {'ALL TESTS PASSED ✅' if all_passed else 'SOME TESTS FAILED ❌'}")
    print("=" * 55)
