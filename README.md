# 💰 MoneyMattersAI

An AI-powered financial intelligence platform.

## 🎯 Current Module: Expense Classification

Categorizes user transactions into spending categories using ML.

| Transaction         | Predicted Category |
|---------------------|--------------------|
| "zomato order"      | Food               |
| "uber ride"         | Transport          |
| "amazon purchase"   | Shopping           |
| "electricity bill"  | Utilities          |

## 🛠 Tech Stack

- **Python** — Core language
- **Pandas** — Data manipulation
- **Scikit-learn** — ML model (TF-IDF + Multinomial Naive Bayes)
- **Joblib** — Model serialization

## 📁 Project Structure

```
MoneyMattersAI/
├── data/
│   └── transactions.csv        # Labeled training data
├── models/
│   └── expense_classifier.pkl  # Trained model (generated)
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Text cleaning utilities
│   ├── train_model.py          # Model training pipeline
│   └── predict.py              # Inference module
├── notebooks/
│   └── experimentation.ipynb   # EDA & experimentation
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.train_model

# Predict a category
python -m src.predict "zomato dinner"
```

## 🗺 Roadmap

- [x] Expense Classification Model
- [ ] Personal Finance Analyzer
- [ ] Stock Market Predictor
- [ ] AI Financial Advisor (Fine-tuned LLM)
