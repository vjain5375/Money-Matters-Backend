"""
MoneyMattersAI — Model Training Module
=======================================

End-to-end training pipeline for the Expense Classification model.

Pipeline
--------
transactions_clean.csv
    → clean_transaction_text()
    → TF-IDF Vectorizer
    → Multinomial Naive Bayes
    → Evaluate
    → Save model + vectorizer

Usage
-----
python src/train_model.py
"""

import os
import sys
import time
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────
# Resolve project root
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_transaction_text

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "transactions_clean.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "expense_classifier.pkl")

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────

def log(step, message):
    print(f"[{step}] {message}")


# ─────────────────────────────────────────────────────────────
# Step 1 — Load Dataset
# ─────────────────────────────────────────────────────────────

def load_dataset(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    required = {"transaction", "category"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset missing columns: {required - set(df.columns)}"
        )

    df = df.dropna(subset=["transaction", "category"])

    return df


# ─────────────────────────────────────────────────────────────
# Step 2 — Preprocess
# ─────────────────────────────────────────────────────────────

def preprocess_transactions(df):

    df = df.copy()

    df["cleaned"] = df["transaction"].apply(clean_transaction_text)

    return df


# ─────────────────────────────────────────────────────────────
# Step 3 — Vectorization
# ─────────────────────────────────────────────────────────────

def vectorize_text(X_train, X_test):

    vectorizer = TfidfVectorizer(
      max_features=2000,
      ngram_range=(1, 2),
      min_df=2,
      sublinear_tf=True
)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer


# ─────────────────────────────────────────────────────────────
# Step 4 — Train Model
# ─────────────────────────────────────────────────────────────

def train_model(X_train_vec, y_train):

    model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)
    model.fit(X_train_vec, y_train)

    return model


# ─────────────────────────────────────────────────────────────
# Step 5 — Evaluate
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test_vec, y_test):

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n===================================================")
    print("MODEL EVALUATION")
    print("===================================================")

    print(f"\nAccuracy: {accuracy:.2%}\n")

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return accuracy


# ─────────────────────────────────────────────────────────────
# Step 6 — Save Model
# ─────────────────────────────────────────────────────────────

def save_artifacts(model, vectorizer):

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model separately (predict.py loads MODEL_PATH and VECTORIZER_PATH separately)
    joblib.dump(model, MODEL_PATH)
    log("SAVE", f"Model saved → {MODEL_PATH}")

    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    log("SAVE", f"Vectorizer saved → {vectorizer_path}")
# ─────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────

def run_training_pipeline():

    print("\n===================================================")
    print("MoneyMattersAI — Expense Classifier Training")
    print("===================================================\n")

    start = time.time()

    # Step 1 — Load
    log("1/7", "Loading dataset")

    df = load_dataset(DATASET_PATH)

    # Normalize categories
    df["category"] = df["category"].astype(str).str.strip().str.title()

    log("1/7", f"Loaded {len(df)} transactions")
    log("1/7", f"Categories: {df['category'].nunique()}")

    # Fix rare categories
    counts = df["category"].value_counts()

    rare = counts[counts < 3].index

    if len(rare) > 0:
        log("WARN", f"Merging {len(rare)} rare categories → 'Other'")
        df.loc[df["category"].isin(rare), "category"] = "Other"

    # Step 2 — Preprocess
    log("2/7", "Cleaning transaction text")

    df = preprocess_transactions(df)

    log("2/7", f"Sample: {df['cleaned'].iloc[0]}")

    # Step 3 — Split
    log("3/7", "Train/Test split")

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"],
        df["category"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"]
    )

    log("3/7", f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 4 — Vectorize
    log("4/7", "TF-IDF vectorization")

    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    log("4/7", f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Step 5 — Train
    log("5/7", "Training model")

    model = train_model(X_train_vec, y_train)

    # Step 6 — Evaluate
    log("6/7", "Evaluating model")

    accuracy = evaluate_model(model, X_test_vec, y_test)

    # Step 7 — Save
    log("7/7", "Saving artifacts")

    save_artifacts(model, vectorizer)

    elapsed = time.time() - start

    print("\nTraining completed")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Time: {elapsed:.2f}s\n")


# ─────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    try:
        run_training_pipeline()

    except Exception as e:
        print("\nERROR:", e)
        sys.exit(1)