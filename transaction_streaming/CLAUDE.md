# CLAUDE.md: Transaction Fraud Report

Context for working on this project. It is one project in Tobi's GitHub Pages portfolio
(`ruxilant360/ruxilant360.github.io`, local `portfolio-site/`). The page is served at
`https://ruxilant360.github.io/transaction_streaming/`, and the home page (`../index.html`) has a card
linking to it.

## What the project is

Fraud analysis of 1,048,575 simulated mobile-money transactions (PaySim, 743 hours, about 31 days), in three parts:

1. **A dashboard page** (`index.html`): KPIs and charts on where fraud happens, a leakage-checked
   XGBoost model, and a form that scores a transaction in the browser.
2. **A Streamlit app** (`python/app.py`): a chat box where a plain-English question is turned into SQL
   by a small free model on **Groq**, run read-only on PostgreSQL and answered in a sentence, plus a fraud prediction form.
3. **The Python scripts** that build the numbers, the sample and the model.

## Rules for this project (from Tobi)

- All files for this project live inside `transaction_streaming/`.
- Use the shared virtual env `portfolio-site/penv` (Windows: `penv\Scripts\python`). Don't create another one.
- Don't commit the big raw files (see `.gitignore`). Commit or push only when Tobi asks.
- The dashboard is **chart-first**. Charts with a short title and a one-line caption that carries the key number.
  No "Question 1–5" cards with long prose.
- The LLM is **Groq** (free tier, `GROQ_API_KEY`), not Anthropic/Claude.
- The Streamlit app stays minimal: one chat box with a placeholder on the Ask tab, and a bare prediction form.
  Don't add text explaining what the model does.
- Teach in plain terms, give the whole picture, and be honest about limits (leakage, simulator artifacts).

## Data

| File | Notes |
|---|---|
| `transaction_streaming_data.csv` | Full PaySim data. 1,048,575 rows, 78 MB. **Git-ignored.** |
| `notebooks/data.csv` | Larger raw copy, 399 MB. **Git-ignored.** |
| `data/sample_transactions.csv` | 20,000 rows: **every** fraud row (1,348) plus 18,652 random normal rows (seed 42), with an added `transaction_id` (`tx000001…`) and sorted by step. Used by the Streamlit app. |
| `sample_data.csv` | **Not** transaction data. It is NHTS travel-survey data that was in the folder, and nothing here uses it. It may belong to the NHTS project. |

Columns: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest,
newbalanceDest, isFlaggedFraud, isFraud`. `step` is the hour since the start, so hour of day = `step % 24`.

**Sample caveat:** fraud is about 6.7% of the sample but 0.13% of the full data. All dashboard numbers come
from the full data. The Streamlit app shows a warning when a result contains a fraud rate.

## What I found (full data)

| Finding | Number |
|---|---|
| Fraud only appears in TRANSFER and CASH_OUT | TRANSFER fraud rate 0.78%, CASH_OUT 0.18% (4.4× gap). PAYMENT, CASH_IN and DEBIT: 0 frauds |
| Big amounts are riskier | 6.87% of TRANSFER/CASH_OUT over $5M were fraud. 47 frauds were exactly $10M, which looks like a per-transaction cap. Median fraud $431,951 vs $170,528 legit |
| Fraud clusters at night | 9.39% of transactions between 2 and 6 a.m. were fraud vs 0.073% in the daytime. Fraud volume is flat across hours while normal traffic drops at night |
| Fraud empties the account | 97.3% of frauds send the sender's entire balance (`amount == oldbalanceOrg`); 0 legit transactions do. 50.8% of frauds go to a receiver with balance 0 before and after vs 0.063% legit. That second signal uses after-the-fact balances, so it is not used in the model |
| The existing rule barely works | `isFlaggedFraud` fired 4 times all month, all real fraud. It caught 4 of 1,348 frauds (0.30%): $12.35M caught, $2.007B missed |

The account-draining signal is a simulator artifact. It is real in practice but never this clean, and the page says so.

## The model (`python/train_model.py`)

- **Rows:** only TRANSFER and CASH_OUT (other types never have fraud and return 0% / Approve).
- **Time-based split:** train `step < 300` (291,709 rows, 609 frauds); test `step >= 300` (165,464 rows, 739 frauds).
- **XGBoost, fixed settings, no search:** `n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
  objective=binary:logistic, eval_metric=aucpr, tree_method=hist, random_state=42`, with `scale_pos_weight = neg/pos`.

### Leakage checks

| Column | Problem | Decision |
|---|---|---|
| `isFlaggedFraud` | Output of the existing rule, only ever 1 on fraud rows | Dropped |
| `newbalanceOrig`, `newbalanceDest` | Balances after the transaction, not known at approval | Dropped |
| `nameOrig`, `nameDest` | Account IDs, so the model would memorise accounts | Dropped |
| `step` | A running clock through the file | Replaced with `hour = step % 24` |
| Split | A random split mixes future and past | Time-based |

**Features used:** `["is_transfer", "amount", "oldbalanceOrg", "oldbalanceDest", "hour"]`.

### Three models on the same split (threshold 0.5)

| Model | Features | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| `all_columns` (leaky) | everything, including after-balances and the flag | 0.896 | 82% | 79% |
| **`leakage_safe` (used)** | the 5 above | **0.854** | **93.5%** | **24.4%** (TP 691, FP 2,140, FN 48) |
| `amount_type_hour` | no balances | 0.320 | — | — |

**Importance (gain share):** sender balance before 44%, amount 22%, is_transfer 15%,
receiver balance before 10%, hour 9%.

### Score bands (test period): the "share that was actually fraud" chart

| Score | Transactions | Fraud | Share fraud |
|---|---|---|---|
| <10% | 156,767 | 15 | 0.01% |
| 10–50% | 5,866 | 33 | 0.6% |
| 50–90% | 1,927 | 82 | 4% |
| 90–99% | 449 | 170 | 38% |
| 99%+ | 455 | 439 | 96% |

The score ranks risk but is **not** a probability: `scale_pos_weight` inflates it. Decisions everywhere are
≥0.9 Hold for review, ≥0.5 Review, otherwise Approve.

### Outputs

- `model/fraud_xgb.json`: the XGBoost model, used by Streamlit.
- `model_web.json` / `model_web.js`: the trees flattened to `[feature, threshold, yes, no, missing]` or `[leaf]`, plus
  bias (logit of `base_score`), metrics, importance and score bands. Used by the dashboard.
- The script asserts that the browser-style tree walk matches XGBoost (difference < 1e-4; measured 2.25e-06).
  The JS uses `Math.fround` comparisons to match XGBoost's float32 behaviour.
- The old `model.json` / `train.ipynb` / `analyze.ipynb` came before this project. The old model used `newbalance*`
  (leaky), so it isn't used.

## Dashboard (`index.html`)

- Uses `../assets/site.css` (shared portfolio styles, light and dark) and Chart.js 4.4.1 from cdnjs.
- **Data loads via `<script src="data.js">` and `<script src="model_web.js">`** (`window.FRAUD_DATA`,
  `window.FRAUD_MODEL`), not `fetch()`. `fetch` is blocked when the page is opened from disk (`file://`), and that
  left every number and chart blank. Keep it this way. The `.json` copies are still written for download.
- **Layout:** hero, then 5 KPI tiles, then a chart grid:
  - fraud rate by type
  - fraud cases by type
  - fraud rate by hour (full width; line plus volume bars on two axes)
  - fraud rate by amount band
  - fraud vs normal transactions
  - frauds caught by the rule (doughnut)
  - fraud value caught vs missed (doughnut)

  Then the model section: model comparison bars, importance bars, fraud share by score band (full width), and
  the leakage table. After that: the "Score a transaction" form (placeholders only, no explanation; it shows
  "Enter an amount to see the score." until an amount is typed), an "Ask the data in plain English" query box (title, one text box, Ask button; no explanatory text), "How I built it" with
  SQL and Python snippets, and downloads.
- **Query box:** runs in the browser on the 20k sample with sql.js 1.10.3 (`sql-asm.js` from cdnjs, no WASM) and
  `data/sample_transactions.js` (`window.FRAUD_SAMPLE`, built by `python/build_sample_js.py`). Both load on first focus via
  `<script>` tags, so it works from `file://`. SELECT/WITH runs as typed (PostgreSQL `::casts` are stripped); plain English is
  matched by keyword to 9 built-in queries (hour, day, flag rule, account draining, biggest frauds, averages, amount band,
  type, totals). No LLM on the page: an API key can't be put in a static site. Shows the table, a bar chart, the sample-rate
  warning and the SQL.
- `.grid > * { min-width: 0; }` stops `<pre>` blocks from causing sideways scroll on phones. Checked at 375 px.
- The GoatCounter script (`freezeram`) is at the bottom, as on every page.

## Streamlit app (`python/`)

| File | Role |
|---|---|
| `app.py` | Two tabs. **Ask the data**: a chat (`st.chat_input` + `st.chat_message`, history in `st.session_state.chat`, shown in a container above the input). Input starting with `SELECT`/`WITH` runs directly; anything else goes to Groq, the SQL runs, on error the model gets one retry with the error message (`llm.fix_sql`), then `llm.answer` writes a 1–2 sentence answer from the first 50 result rows. Each reply shows the answer, the table, an auto bar chart for 2-column results, a sample-rate warning and the SQL in an expander. **Predict fraud**: 5 inputs with placeholders, then the score, a progress bar and the decision. |
| `db.py` | PostgreSQL via `psycopg`. Everything lives in its own **`fraud` schema** (`fraud.transactions`); `connect()` sets `search_path` to `fraud`, so queries just say `transactions`. Tobi's `postgres` database already has an unrelated `public.transactions` table (columns `transaction_id, account_id, transaction_date, amount, category`, 1,060 rows, next to `accounts`, `ads`, `events`, `platforms`, `videos`): never touch or rename it. `setup()` creates the schema and table (lower-case column names) and `COPY`s the sample in if it is empty. `run()` executes one statement in a `READ ONLY` transaction with a 15 s `statement_timeout`, then rolls back. It rejects multiple statements and converts `Decimal` results to float. |
| `llm.py` | Groq text-to-SQL via the `groq` SDK (OpenAI-style `client.chat.completions.create`). `model = GROQ_MODEL` or **`openai/gpt-oss-20b`** (small, fast, free tier), `temperature=0`; for gpt-oss models it sends `reasoning_effort="low"` and `include_reasoning=False` so the reply is only the answer, and it strips any `<think>` block (Qwen). `SYSTEM` prompt with the schema, the sample caveat and PostgreSQL rules (`ROUND(x::numeric, n)`, avoid integer division); it extracts the ```sql block (or bare SELECT), and a reply with no SQL is shown as the "can't answer" message. `ANSWER_SYSTEM` + `answer()` turn the result CSV into a sentence, saying "in this sample" for fraud rates. The key is read with `.strip().strip('"')` because `.env` had `GROQ_API_KEY= "gsk_..."`. |
| `.env.example` | `GROQ_API_KEY=` (free from console.groq.com/keys), `DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/postgres`, optional `GROQ_MODEL`. Copy it to `.env`, which is git-ignored. |
| `requirements.txt` | pandas, numpy, scikit-learn, xgboost, streamlit, groq, python-dotenv, psycopg[binary] |

**Groq models** available to this key on 2026-11-09: `openai/gpt-oss-20b` (default), `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`,
`allam-2-7b`, plus non-chat models (whisper, orpheus, prompt-guard). The Llama 3.x chat models are no longer listed.
List them with `GET https://api.groq.com/openai/v1/models` and the key as a Bearer token.

**Slow imports on this PC:** xgboost about 26 s, pandas with Streamlit about 14 s. Importing them at
the top froze the app for close to a minute: the text box ignored typing and Score did nothing. They are now
imported lazily (`get_model()`, `llm._get_client()`) and xgboost and groq are warmed up in a background thread (`warm_up()`).
Don't move these imports back to the top.

**Status (2026-11-09):** tested end to end with Tobi's `.env`: PostgreSQL 18 runs locally as a Windows service, the
sample loaded into `fraud.transactions` (20,000 rows), writes are blocked, and Groq answered questions in about 0.3–3 s.
Off-topic questions ("weather in Lagos") get a one-line refusal. The chat box was driven with Streamlit's `AppTest`:
a question gets an answer, table and warning, typed SQL runs, and history is kept.

## How to run (from `portfolio-site/`)

```
penv\Scripts\pip install -r transaction_streaming\python\requirements.txt
penv\Scripts\python transaction_streaming\python\build_data.py    # needs transaction_streaming_data.csv; writes data.json, data.js, the sample
penv\Scripts\python transaction_streaming\python\build_sample_js.py   # writes data/sample_transactions.js for the page's query box
penv\Scripts\python transaction_streaming\python\train_model.py   # writes model/fraud_xgb.json, model_web.json, model_web.js
penv\Scripts\streamlit run transaction_streaming\python\app.py
```

Local previews (in `DATA ANALYTICS PRACTICE/.claude/launch.json`): `portfolio` serves `portfolio-site` on port 8765
(the dashboard is at `/transaction_streaming/`), and `fraud-app` runs Streamlit on port 8501.

## Git

- `transaction_streaming/.gitignore` excludes the two big CSVs, `notebooks/.ipynb_checkpoints/`, `python/.env` and `__pycache__/`.
- The root `.gitignore` excludes `penv/`. It must be ASCII, not UTF-16 (PowerShell `echo` wrote UTF-16 once and git ignored the file).
- To publish: `git add index.html transaction_streaming`, commit, `git pull --rebase`, `git push`. Decide first whether
  `sample_data.csv` (NHTS) belongs here.

## Change log

1. Found that `sample_data.csv` is NHTS data. Built a 20k transaction sample instead, keeping all fraud rows.
2. `build_data.py`: KPIs and the findings above from the full data, plus the sample.
3. `train_model.py`: leakage review, time split, 3-model comparison, tree export with a parity check.
4. First dashboard (question cards) and the Streamlit app (SQLite, editable SQL, schema tab). Added the home-page card.
5. Rewrite at Tobi's request: chart-first dashboard with 10 charts; data moved from `fetch` to `.js` files so it renders
   from disk; Streamlit cut to a text box and a bare form.
6. Fixed the frozen text box and missing score in Streamlit (lazy imports plus background warm-up). The dashboard form
   prompts for an amount.
7. The app was switched from in-memory SQLite to local PostgreSQL (`db.py`, `llm.py` prompt, `psycopg`), with
   direct SELECT input, a one-retry SQL fix and a sample-rate warning. README updated to match.
8. The dashboard's Ask card was static text with no input. Replaced it with a working in-browser query box and removed the description text.
9. Switched the LLM from Anthropic to Groq (`openai/gpt-oss-20b`). Moved the app's table to a `fraud` schema because `public.transactions` belonged to another project. The Ask tab became a chat with a one-sentence answer from the result.
