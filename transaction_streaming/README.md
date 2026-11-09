# Transaction Fraud Report

Fraud analysis of 1,048,575 simulated mobile-money transactions (PaySim), a leakage-checked
XGBoost model, and a Streamlit app that turns plain-English questions into SQL (and a one-sentence answer) with a free small model on Groq.

Live page: https://ruxilant360.github.io/transaction_streaming/

## What's here

| Path | What it is |
|---|---|
| `index.html`, `data.js`, `model_web.js` | The dashboard page, its numbers, and the model trees it scores in the browser (`.json` copies of the same data are kept for download) |
| `data/sample_transactions.csv` | 20,000-row sample: every fraud row (1,348) + random normal rows. Used by the Streamlit app |
| `model/fraud_xgb.json` | The trained leakage-safe XGBoost model |
| `python/build_data.py` | Answers the 5 dashboard questions and writes the sample |
| `python/train_model.py` | Trains the model, runs the leakage comparison, exports the trees |
| `python/app.py`, `db.py`, `llm.py` | The Streamlit app: Groq text-to-SQL chat, read-only PostgreSQL queries on the sample, prediction form |
| `train.ipynb`, `analyze.ipynb`, `model.json` | Earlier notebooks and model (the old model used after-the-fact balances, so it isn't used here) |

The full data files (`transaction_streaming_data.csv`, `notebooks/data.csv`) are too large for GitHub
and are not committed. They come from the PaySim dataset on Kaggle.

## Run it (Windows, from the `portfolio-site` folder)

```
penv\Scripts\pip install -r transaction_streaming\python\requirements.txt
penv\Scripts\python transaction_streaming\python\build_data.py     # needs transaction_streaming_data.csv
penv\Scripts\python transaction_streaming\python\train_model.py
penv\Scripts\streamlit run transaction_streaming\python\app.py
```

Copy `python/.env.example` to `python/.env`, set `DATABASE_URL` for your local PostgreSQL
(the app creates and fills a `fraud.transactions` table on first run), and add your free
`GROQ_API_KEY` from https://console.groq.com/keys. Without a key the app still runs, and you can write SQL yourself.

## Leakage checks

- `isFlaggedFraud`: output of the existing rule, only ever 1 on fraud rows. Dropped.
- `newbalanceOrig`, `newbalanceDest`: balances after the transaction. Dropped.
- `nameOrig`, `nameDest`: account IDs. Dropped.
- `step`: replaced with hour of day; train on hours 1–299, test on 300–743.
