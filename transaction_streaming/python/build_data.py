"""Answer the 5 dashboard questions and write the 20,000-row sample.

Reads the full transaction file (transaction_streaming_data.csv, kept out of git),
writes:
  - data/sample_transactions.csv  20,000 rows: every fraud row + a random sample of normal rows
  - data.json                     the numbers the dashboard page shows

Usage (from the portfolio-site folder):
    penv\\Scripts\\python transaction_streaming\\python\\build_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # transaction_streaming/
FULL = ROOT / "transaction_streaming_data.csv"
SAMPLE = ROOT / "data" / "sample_transactions.csv"
OUT = ROOT / "data.json"
SAMPLE_ROWS = 20_000
SEED = 42

df = pd.read_csv(FULL)
df["hour"] = df.step % 24                    # step = hours since the simulation started
df["drains_account"] = np.isclose(df.amount, df.oldbalanceOrg) & (df.amount > 0)
n, fraud = len(df), df[df.isFraud == 1]
risky = df[df.type.isin(["TRANSFER", "CASH_OUT"])]   # the only types where fraud occurs

# ---------------------------------------------------------------- sample for the SQL app
normal = df[df.isFraud == 0].sample(SAMPLE_ROWS - len(fraud), random_state=SEED)
sample = (pd.concat([fraud, normal]).sort_values("step").reset_index(drop=True)
          .drop(columns=["hour", "drains_account"]))
sample.insert(0, "transaction_id", ["tx" + str(i).zfill(6) for i in range(1, len(sample) + 1)])
SAMPLE.parent.mkdir(exist_ok=True)
sample.to_csv(SAMPLE, index=False)

# ---------------------------------------------------------------- KPIs
kpi = {
    "transactions": n,
    "value": round(df.amount.sum(), 2),
    "fraud_count": len(fraud),
    "fraud_value": round(fraud.amount.sum(), 2),
    "fraud_rate": len(fraud) / n,
    "steps": int(df.step.max()),
}

# ---------------------------------------------------------------- Q1: which transaction types carry fraud?
q1 = (df.groupby("type")
      .agg(count=("amount", "size"), value=("amount", "sum"),
           fraud=("isFraud", "sum"),
           fraud_value=("amount", lambda s: s[df.loc[s.index, "isFraud"] == 1].sum()))
      .sort_values("count", ascending=False).reset_index())
q1["fraud_rate"] = q1.fraud / q1["count"]

# ---------------------------------------------------------------- Q2: how big are fraudulent transfers?
bands = [0, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, np.inf]
labels = ["Under $10K", "$10K–100K", "$100K–500K", "$500K–1M", "$1M–5M", "$5M+"]
risky = risky.assign(band=pd.cut(risky.amount, bands, labels=labels, right=False))
q2 = (risky.groupby("band", observed=False)
      .agg(count=("amount", "size"), fraud=("isFraud", "sum")).reset_index())
q2["fraud_rate"] = q2.fraud / q2["count"]
capped = int((fraud.amount == 10_000_000).sum())     # fraud amounts stop at exactly $10M

# ---------------------------------------------------------------- Q3: when does fraud happen?
q3 = df.groupby("hour").agg(count=("amount", "size"), fraud=("isFraud", "sum")).reset_index()
q3["fraud_rate"] = q3.fraud / q3["count"]
night = q3[q3.hour.between(2, 6)]

# ---------------------------------------------------------------- Q4: do fraudsters empty the account?
legit_risky = risky[risky.isFraud == 0]
q4 = {
    "fraud_drain_share": float(fraud.drains_account.mean()),
    "legit_drain_share": float(legit_risky.drains_account.mean()),
    "fraud_dest_zero": float(((fraud.oldbalanceDest == 0) & (fraud.newbalanceDest == 0)).mean()),
    "legit_dest_zero": float(((legit_risky.oldbalanceDest == 0) & (legit_risky.newbalanceDest == 0)).mean()),
}

# ---------------------------------------------------------------- Q5: how much does the existing rule catch?
flagged = df[df.isFlaggedFraud == 1]
q5 = {
    "flagged": len(flagged),
    "flagged_and_fraud": int(flagged.isFraud.sum()),
    "recall": float(flagged.isFraud.sum() / len(fraud)),
    "value_caught": round(flagged[flagged.isFraud == 1].amount.sum(), 2),
    "value_missed": round(fraud.amount.sum() - flagged[flagged.isFraud == 1].amount.sum(), 2),
}

out = {
    "kpi": kpi,
    "q1_types": q1.round(6).to_dict(orient="records"),
    "q2_bands": q2.assign(band=q2.band.astype(str)).round(6).to_dict(orient="records"),
    "q2_capped_10m": capped,
    "q2_fraud_median": float(fraud.amount.median()),
    "q2_legit_median": float(legit_risky.amount.median()),
    "q3_hours": q3.round(6).to_dict(orient="records"),
    "q3_night_rate": float(night.fraud.sum() / night["count"].sum()),
    "q3_night_share": float(night.fraud.sum() / len(fraud)),
    "q3_day_rate": float(q3[q3.hour.between(10, 20)].fraud.sum() / q3[q3.hour.between(10, 20)]["count"].sum()),
    "q4": q4,
    "q5": q5,
    "sample_rows": len(sample),
    "sample_fraud": int(sample.isFraud.sum()),
}
OUT.write_text(json.dumps(out, separators=(",", ":")))
# same numbers as a script, so the page also works when opened straight from disk
(ROOT / "data.js").write_text("window.FRAUD_DATA=" + json.dumps(out, separators=(",", ":")) + ";")
print(f"wrote {OUT.name} and {SAMPLE.relative_to(ROOT)} ({len(sample):,} rows, {sample.isFraud.sum():,} fraud)")
print(json.dumps({k: out[k] for k in ["kpi", "q4", "q5", "q3_night_rate", "q3_night_share", "q3_day_rate"]}, indent=1))
