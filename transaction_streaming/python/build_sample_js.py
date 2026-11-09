r"""Write data/sample_transactions.js (window.FRAUD_SAMPLE) from the 20k sample CSV.

The dashboard's query box loads it with a <script> tag, because fetch() is blocked when the page is opened from disk.
Run after build_data.py:  penv\Scripts\python transaction_streaming\python\build_sample_js.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "sample_transactions.csv")
df.columns = [c.lower() for c in df.columns]
out = {"columns": list(df.columns), "rows": df.values.tolist()}
(ROOT / "data" / "sample_transactions.js").write_text(
    "window.FRAUD_SAMPLE=" + json.dumps(out, separators=(",", ":")) + ";\n", encoding="utf-8")
print(len(df), "rows ->", ROOT / "data" / "sample_transactions.js")
