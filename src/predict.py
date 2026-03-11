"""
MoneyMattersAI -- Prediction Module
====================================

This module loads the trained Expense Classification model and TF-IDF
vectorizer, then classifies new transaction descriptions into spending
categories.

Pipeline:
    User input
        -> clean_transaction_text()
        -> vectorizer.transform()
        -> model.predict() + model.predict_proba()
        -> Structured result dict

Usage (CLI):
    python src/predict.py "zomato dinner"
    python src/predict.py "uber ride to airport"

Usage (Import):
    from src.predict import predict_transaction
    result = predict_transaction("zomato dinner")
    print(result["predicted_category"])  # "Food"
"""

import os
import sys
import json
import argparse
import numpy as np
import joblib

# -- Resolve project root so the script works from any directory ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_transaction_text


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "expense_classifier.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")


# -------------------------------------------------------------------------
# Helper: pretty logger
# -------------------------------------------------------------------------
def _log(step: str, message: str) -> None:
    """Print a formatted log line with a step label."""
    print(f"  [{step}]  {message}")


# -------------------------------------------------------------------------
# Artifact Loader (with caching)
# -------------------------------------------------------------------------
_cached_model = None
_cached_vectorizer = None


def load_artifacts(verbose: bool = True):
    """
    Load the trained model and TF-IDF vectorizer from disk.

    Artifacts are cached in module-level variables so repeated calls
    (e.g., in a REST API loop) do not re-read from disk every time.

    Args:
        verbose: If True, print loading progress.

    Returns:
        Tuple of (model, vectorizer).

    Raises:
        FileNotFoundError: If model or vectorizer .pkl files are missing.
    """
    global _cached_model, _cached_vectorizer

    # Return cached artifacts if already loaded
    if _cached_model is not None and _cached_vectorizer is not None:
        return _cached_model, _cached_vectorizer

    # -- Validate that artifact files exist --------------------------------
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}\n"
            f"Please run 'python src/train_model.py' first to train the model."
        )

    if not os.path.isfile(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at: {VECTORIZER_PATH}\n"
            f"Please run 'python src/train_model.py' first to train the model."
        )

    # -- Load artifacts ----------------------------------------------------
    if verbose:
        _log("LOAD", "Loading trained model...")
    _cached_model = joblib.load(MODEL_PATH)

    if verbose:
        _log("LOAD", "Loading TF-IDF vectorizer...")
    _cached_vectorizer = joblib.load(VECTORIZER_PATH)

    if verbose:
        _log("LOAD", f"Model classes: {list(_cached_model.classes_)}")
        _log("LOAD", f"Vectorizer vocabulary size: {len(_cached_vectorizer.vocabulary_)}")

    return _cached_model, _cached_vectorizer


# -------------------------------------------------------------------------
# Core Prediction Function
# -------------------------------------------------------------------------
def predict_transaction(text: str, verbose: bool = True) -> dict:
    """
    Predict the expense category for a single transaction description.

    This function runs the full inference pipeline:
        1. Validate input
        2. Clean text using clean_transaction_text()
        3. Vectorize using the trained TF-IDF vectorizer
        4. Predict category using the trained Naive Bayes model
        5. Compute confidence score via predict_proba()

    Args:
        text:    Raw transaction description (e.g., "Zomato Order #123").
        verbose: If True, print step-by-step logs for debugging.

    Returns:
        A structured dictionary ready for JSON serialization or LLM input:
        {
            "transaction": "Zomato Order #123",
            "cleaned_text": "zomato order",
            "predicted_category": "Food",
            "confidence": 0.92,
            "all_probabilities": {
                "Food": 0.92,
                "Shopping": 0.03,
                "Transport": 0.02,
                "Utilities": 0.03
            }
        }

    Raises:
        ValueError: If input text is None, not a string, or empty.
        FileNotFoundError: If model artifacts are not found.
    """

    # -- Step 0: Validate input -------------------------------------------
    if text is None:
        raise ValueError("Input text cannot be None.")
    if not isinstance(text, str):
        raise ValueError(f"Expected a string, got {type(text).__name__}.")
    if text.strip() == "":
        raise ValueError("Input text cannot be empty.")

    # -- Step 1: Load model and vectorizer --------------------------------
    model, vectorizer = load_artifacts(verbose=verbose)

    # -- Step 2: Clean the transaction text --------------------------------
    if verbose:
        _log("CLEAN", f"Raw input: \"{text}\"")

    cleaned = clean_transaction_text(text)

    if verbose:
        _log("CLEAN", f"Cleaned:   \"{cleaned}\"")

    # Guard against text that becomes empty after cleaning
    if cleaned == "":
        raise ValueError(
            f"Input text \"{text}\" became empty after preprocessing. "
            f"Please provide a meaningful transaction description."
        )

    # -- Step 3: Vectorize using trained TF-IDF ----------------------------
    if verbose:
        _log("TFIDF", "Vectorizing cleaned text...")

    try:
        text_vector = vectorizer.transform([cleaned])
    except Exception as e:
        raise RuntimeError(f"Vectorization failed: {e}")

    if verbose:
        _log("TFIDF", f"Feature vector shape: {text_vector.shape}")

    # -- Step 4: Predict category ------------------------------------------
    if verbose:
        _log("PRED", "Predicting category...")

    predicted_category = model.predict(text_vector)[0]

    # -- Step 5: Compute confidence via predict_proba() --------------------
    # predict_proba() returns a probability distribution over all classes.
    # The confidence score is the maximum probability -- i.e., how sure the
    # model is about its top prediction.
    probabilities = model.predict_proba(text_vector)[0]
    confidence = float(np.max(probabilities))

    # Build a readable probability map: {"Food": 0.92, "Transport": 0.03, ...}
    class_labels = list(model.classes_)
    all_probabilities = {
        label: round(float(prob), 4)
        for label, prob in zip(class_labels, probabilities)
    }

    # -- Build structured result -------------------------------------------
    result = {
        "transaction": text,
        "cleaned_text": cleaned,
        "predicted_category": str(predicted_category),
        "confidence": round(confidence, 4),
        "all_probabilities": all_probabilities,
    }

    if verbose:
        _log("PRED", f"Category: {result['predicted_category']} "
                      f"(confidence: {result['confidence']:.2%})")

    return result


# -------------------------------------------------------------------------
# Batch Prediction (for future API/pipeline use)
# -------------------------------------------------------------------------
def predict_batch(transactions: list, verbose: bool = False) -> list:
    """
    Predict categories for a list of transaction descriptions.

    Useful for bulk processing in an API endpoint or data pipeline.

    Args:
        transactions: List of raw transaction strings.
        verbose: If True, print logs for each prediction.

    Returns:
        List of structured result dictionaries.
    """
    results = []
    for text in transactions:
        try:
            result = predict_transaction(text, verbose=verbose)
            results.append(result)
        except (ValueError, RuntimeError) as e:
            results.append({
                "transaction": text,
                "error": str(e),
            })
    return results


# -------------------------------------------------------------------------
# Pretty Print Result
# -------------------------------------------------------------------------
def _print_result(result: dict) -> None:
    """Display a prediction result in a user-friendly format."""
    print()
    print("=" * 55)
    print("  MoneyMattersAI -- Expense Prediction Result")
    print("=" * 55)
    print()
    print(f"  Transaction:        {result['transaction']}")
    print(f"  Cleaned Text:       {result['cleaned_text']}")
    print(f"  Predicted Category: {result['predicted_category']}")
    print(f"  Confidence:         {result['confidence']:.2%}")
    print()

    # Show probability breakdown
    print("  Probability Breakdown:")
    print("  " + "-" * 35)
    for category, prob in sorted(
        result["all_probabilities"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        bar = "#" * int(prob * 30)
        marker = " <<" if category == result["predicted_category"] else ""
        print(f"    {category:<12}  {prob:>6.2%}  {bar}{marker}")

    print()
    print("  JSON Output (for LLM pipeline):")
    print("  " + "-" * 35)

    # Build the clean JSON output for LLM integration
    json_output = {
        "transaction": result["transaction"],
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
    }
    print(f"  {json.dumps(json_output, indent=2)}")
    print()
    print("=" * 55)
    print()


# -------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------
def main():
    """
    Command-line interface for single transaction prediction.

    Usage:
        python src/predict.py "zomato dinner"
        python src/predict.py "uber ride to airport"
        python src/predict.py --batch "zomato dinner" "uber ride" "amazon order"
    """
    parser = argparse.ArgumentParser(
        prog="MoneyMattersAI Predict",
        description="Predict expense category for a transaction description.",
    )
    parser.add_argument(
        "transaction",
        nargs="+",
        help="Transaction description(s) to classify. "
             "Pass multiple for batch prediction.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as raw JSON (useful for piping to other tools).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step debug logs.",
    )

    args = parser.parse_args()

    try:
        if len(args.transaction) == 1:
            # -- Single prediction -----------------------------------------
            result = predict_transaction(
                args.transaction[0],
                verbose=not args.quiet,
            )

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                _print_result(result)

        else:
            # -- Batch prediction ------------------------------------------
            results = predict_batch(
                args.transaction,
                verbose=not args.quiet,
            )

            if args.json:
                print(json.dumps(results, indent=2))
            else:
                for result in results:
                    if "error" in result:
                        print(f"\n  [ERROR] \"{result['transaction']}\": {result['error']}")
                    else:
                        _print_result(result)

    except FileNotFoundError as e:
        print(f"\n  [ERROR] MODEL NOT FOUND: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n  [ERROR] INVALID INPUT: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] UNEXPECTED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
