import pandas as pd
import os

# Resolve project root automatically
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "transactions_clean.csv")


def load_all_datasets():
    """Load all CSV and Excel datasets from raw folder"""

    dfs = []

    for file in os.listdir(RAW_DIR):

        path = os.path.join(RAW_DIR, file)

        print(f"\nLoading: {file}")

        try:
            if file.endswith(".csv"):
                df = pd.read_csv(path)

            elif file.endswith(".xlsx"):
                df = pd.read_excel(path)

            else:
                continue

            print("Columns:", list(df.columns))

            dfs.append(df)

        except Exception as e:
            print(f"Error loading {file}: {e}")

    return dfs


def normalize_columns(df):

    column_map = {
        "Description": "transaction",
        "description": "transaction",
        "merchant": "transaction",
        "Merchant": "transaction",
        "Title": "transaction",
        "Name": "transaction",

        "Amount": "amount",
        "amount": "amount",

        "Category": "category",
        "category": "category"
    }

    df = df.rename(columns=column_map)

    # If multiple transaction columns exist, merge them
    transaction_cols = [col for col in df.columns if col == "transaction"]

    if len(transaction_cols) > 1:
        df["transaction"] = df[transaction_cols].astype(str).agg(" ".join, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def clean_dataset(dfs):
    """Clean and merge datasets"""

    cleaned = []

    for df in dfs:

        df = normalize_columns(df)

        # If multiple 'transaction' columns exist, merge them
        if df.columns.tolist().count("transaction") > 1:
            transaction_cols = [col for col in df.columns if col == "transaction"]
            df["transaction"] = df[transaction_cols].astype(str).agg(" ".join, axis=1)
            df = df.loc[:, ~df.columns.duplicated()]

        required_cols = {"transaction", "amount", "category"}

        if not required_cols.issubset(df.columns):
            print("Skipping dataset due to missing required columns")
            continue

        df = df[["transaction", "amount", "category"]]

        # Remove missing values
        df = df.dropna()

        # Ensure correct data types
        df["transaction"] = df["transaction"].astype(str)
        df["transaction"] = df["transaction"].str.lower()

        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        df = df.dropna()

        cleaned.append(df)

    return pd.concat(cleaned, ignore_index=True)


def main():

    print("\n=== Preparing Dataset ===")

    datasets = load_all_datasets()

    df = clean_dataset(datasets)

    # Remove duplicates
    df = df.drop_duplicates()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset prepared successfully!")
    print("Total rows:", len(df))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()