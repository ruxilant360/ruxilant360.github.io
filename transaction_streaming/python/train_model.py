"""Train the XGBoost isFraud model and check that it doesn't cheat.

Leakage checks:
  1. isFlaggedFraud is the output of the bank's existing rule. It is only set on fraud rows,
     so a model that sees it is reading part of the answer. Dropped.
  2. newbalanceOrig / newbalanceDest are balances AFTER the transaction has gone through.
     A model that scores a transaction before approving it can't know them. Dropped.
  3. nameOrig / nameDest are account IDs: memorising IDs doesn't generalise. Dropped.
  4. step is a running clock: raw step lets the model learn "when" in this file. Replaced
     with hour of day, and the split is by time (train on earlier hours, test on later ones),
     so the model is always tested on transactions that happen after the ones it learned from.

Three models are trained on the same split to show the effect:
  A. all columns (what you get if you don't check)
  B. leakage-safe (the model the dashboard uses)
  C. amount, type and hour only (no balances at all)

Outputs (in transaction_streaming/):
  model/fraud_xgb.json   the leakage-safe model, for the Streamlit app
  model_web.json         the same trees in a compact form + metrics, for the dashboard page

Usage (from the portfolio-site folder):
    penv\\Scripts\\python transaction_streaming\\python\\train_model.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "transaction_streaming_data.csv"
MODEL_DIR = ROOT / "model"
SPLIT_STEP = 300          # train on hours 1–299, test on hours 300–743
THRESHOLD = 0.5

SAFE = ["is_transfer", "amount", "oldbalanceOrg", "oldbalanceDest", "hour"]
FEATURE_LABELS = {
    "is_transfer": "Type is TRANSFER (vs CASH_OUT)",
    "amount": "Amount",
    "oldbalanceOrg": "Sender balance before",
    "oldbalanceDest": "Receiver balance before",
    "hour": "Hour of day",
}
VARIANTS = {
    "all_columns": ["is_transfer", "amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest", "isFlaggedFraud", "step"],
    "leakage_safe": SAFE,
    "amount_type_hour": ["is_transfer", "amount", "hour"],
}
PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
              objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
              random_state=42, n_jobs=-1)

# ---------------------------------------------------------------- data
df = pd.read_csv(FULL)
df = df[df.type.isin(["TRANSFER", "CASH_OUT"])].copy()      # fraud only occurs in these two types
df["is_transfer"] = (df.type == "TRANSFER").astype(int)
df["hour"] = df.step % 24
train, test = df[df.step < SPLIT_STEP], df[df.step >= SPLIT_STEP]
y_tr, y_te = train.isFraud, test.isFraud
spw = (y_tr == 0).sum() / (y_tr == 1).sum()                 # balance the rare fraud class
print(f"train {len(train):,} rows ({y_tr.sum()} fraud) | test {len(test):,} rows ({y_te.sum()} fraud)")


def evaluate(model, cols):
    p = model.predict_proba(test[cols])[:, 1]
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return {
        "pr_auc": round(average_precision_score(y_te, p), 4),
        "roc_auc": round(roc_auc_score(y_te, p), 4),
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


results, models = {}, {}
for name, cols in VARIANTS.items():
    m = xgb.XGBClassifier(scale_pos_weight=spw, **PARAMS).fit(train[cols], y_tr)
    results[name] = evaluate(m, cols)
    models[name] = m
    print(f"{name:18s} {results[name]}")

safe = models["leakage_safe"]
gain = safe.get_booster().get_score(importance_type="gain")
total = sum(gain.values())
importance = sorted(({"feature": f, "label": FEATURE_LABELS[f], "share": round(gain.get(f, 0) / total, 4)}
                     for f in SAFE), key=lambda d: -d["share"])

# Fraud share by score band on the test set (shows the score works as a ranking)
p_te = safe.predict_proba(test[SAFE])[:, 1]
bands = pd.cut(p_te, [0, .1, .5, .9, .99, 1.0001], labels=["<10%", "10–50%", "50–90%", "90–99%", "99%+"], right=False)
score_bands = (pd.DataFrame({"band": bands, "fraud": y_te.values})
               .groupby("band", observed=False).fraud.agg(["size", "sum"]).reset_index()
               .rename(columns={"size": "count", "sum": "fraud"}))

# ---------------------------------------------------------------- save the model for Streamlit
MODEL_DIR.mkdir(exist_ok=True)
safe.save_model(MODEL_DIR / "fraud_xgb.json")

# ---------------------------------------------------------------- export trees for the web page
booster = safe.get_booster()
fidx = {f: i for i, f in enumerate(SAFE)}


def flatten(node, out):
    """Tree -> flat list. Split: [feature, threshold, yes, no, missing]. Leaf: [value]."""
    i = len(out)
    out.append(None)
    if "leaf" in node:
        out[i] = [round(node["leaf"], 6)]
        return i
    kids = {c["nodeid"]: c for c in node["children"]}
    y = flatten(kids[node["yes"]], out)
    n = flatten(kids[node["no"]], out)
    miss = y if node["missing"] == node["yes"] else n
    out[i] = [fidx[node["split"]], node["split_condition"], y, n, miss]
    return i


trees = []
for t in booster.get_dump(dump_format="json"):
    flat = []
    flatten(json.loads(t), flat)
    trees.append(flat)
base_score = float(json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))
bias = float(np.log(base_score / (1 - base_score)))


def score_like_browser(row):
    """Walk the exported trees exactly as the dashboard's JavaScript does."""
    margin = bias
    for t in trees:
        k = 0
        while len(t[k]) > 1:
            f, thr, y, n, miss = t[k]
            v = row[f]
            k = miss if v is None or np.isnan(v) else (y if np.float32(v) < np.float32(thr) else n)
        margin += t[k][0]
    return 1 / (1 + np.exp(-margin))


check = test[SAFE].sample(3000, random_state=1)
diff = np.abs(np.array([score_like_browser(r) for r in check.to_numpy(dtype=float)])
              - safe.predict_proba(check)[:, 1]).max()
print(f"browser scorer matches XGBoost: max difference {diff:.2e}")
assert diff < 1e-4, "exported trees don't reproduce the model"

web = {
    "features": SAFE,
    "feature_labels": FEATURE_LABELS,
    "bias": round(bias, 6),
    "trees": trees,
    "threshold": THRESHOLD,
    "split_step": SPLIT_STEP,
    "train_rows": len(train), "train_fraud": int(y_tr.sum()),
    "test_rows": len(test), "test_fraud": int(y_te.sum()),
    "params": {k: PARAMS[k] for k in ["n_estimators", "max_depth", "learning_rate", "subsample"]},
    "scale_pos_weight": round(float(spw), 1),
    "results": results,
    "importance": importance,
    "score_bands": score_bands.to_dict(orient="records"),
}
(ROOT / "model_web.json").write_text(json.dumps(web, separators=(",", ":")))
(ROOT / "model_web.js").write_text("window.FRAUD_MODEL=" + json.dumps(web, separators=(",", ":")) + ";")
print(f"wrote model/fraud_xgb.json and model_web.json ({(ROOT / 'model_web.json').stat().st_size / 1024:.0f} KB)")
print(json.dumps(importance, indent=1))
print(score_bands.to_string(index=False))
