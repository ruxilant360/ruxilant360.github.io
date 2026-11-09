"""Streamlit app: ask the transaction data a question in plain English, or score a transaction.

Run (from the portfolio-site folder):
    penv\\Scripts\\streamlit run transaction_streaming\\python\\app.py
"""
import re
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

import db
import llm

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "model" / "fraud_xgb.json"
FEATURES = ["is_transfer", "amount", "oldbalanceOrg", "oldbalanceDest", "hour"]

st.set_page_config(page_title="Transaction Fraud Explorer", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Connecting to PostgreSQL...")
def setup_db() -> int:
    """Create the transactions table and load the sample once per app process."""
    return db.setup()


@st.cache_resource
def warm_up():
    """xgboost and groq take a long time to import on some PCs. Load them in the
    background so the page shows (and the text box works) straight away."""
    def load():
        import xgboost  # noqa: F401
        import groq  # noqa: F401
    threading.Thread(target=load, daemon=True).start()
    return True


@st.cache_resource
def get_model():
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model


def show_result(result: pd.DataFrame, sql: str) -> None:
    if result.empty:
        st.caption("No rows.")
        with st.expander("SQL"):
            st.code(sql, language="sql")
        return
    st.caption(f"{len(result):,} row(s)")
    st.dataframe(result, width="stretch", hide_index=True)
    numeric = result.select_dtypes("number").columns
    if len(result.columns) == 2 and len(numeric) == 1 and 1 < len(result) <= 50:
        st.bar_chart(result.set_index(result.columns[0])[numeric[0]])
    if any("fraud" in c.lower() and ("rate" in c.lower() or "pct" in c.lower() or "percent" in c.lower())
           for c in result.columns):
        st.info("Fraud makes up ~6.7% of this sample (every fraud row was kept) vs 0.13% in the full data, "
                "so rates here are inflated. Compare them with each other, not with the real world.")
    with st.expander("SQL"):
        st.code(sql, language="sql")


warm_up()
st.title("Transaction Fraud Explorer")
ask_tab, predict_tab = st.tabs(["Ask the data", "Predict fraud"])

# ---------------------------------------------------------------- ask in plain English
with ask_tab:
    db_ready = False
    try:
        rows = setup_db()
        db_ready = True
        st.caption(f"PostgreSQL table `transactions`: {rows:,} rows")
    except Exception as e:
        st.error(f"Can't reach PostgreSQL: {e}\n\nSet DATABASE_URL in transaction_streaming/python/.env "
                 "(e.g. postgresql://postgres:PASSWORD@localhost:5432/postgres) and restart the app.")

    if "chat" not in st.session_state:
        st.session_state.chat = []               # [{"role", "text", "kind", "sql", "result"}]

    def render(msg: dict) -> None:
        with st.chat_message(msg["role"]):
            if msg.get("kind") == "error":
                st.error(msg["text"])
            elif msg.get("kind") == "warning":
                st.warning(msg["text"])
            elif msg["text"]:
                st.markdown(msg["text"])
            if msg.get("result") is not None:
                show_result(msg["result"], msg["sql"])
            elif msg.get("sql"):
                with st.expander("SQL"):
                    st.code(msg["sql"], language="sql")

    def respond(question: str) -> dict:
        """Turn the question into SQL (or take it as SQL), run it, and answer it in a sentence."""
        if re.match(r"(?is)^\s*(select|with)\b", question):      # the user typed SQL: run it directly
            try:
                return {"text": "", "sql": question, "result": db.run(question)}
            except Exception as e:
                return {"kind": "error", "text": f"Query failed: {e}", "sql": question}
        if not llm.is_configured():
            return {"kind": "error", "text": "Add GROQ_API_KEY to transaction_streaming/python/.env and restart the app "
                                             "(or type a SELECT query yourself)."}
        try:
            sql, raw = llm.question_to_sql(question)
        except Exception as e:                   # show API problems instead of crashing the app
            return {"kind": "error", "text": f"Groq request failed: {e}"}
        if not sql:
            return {"kind": "warning", "text": raw}
        try:
            result = db.run(sql)
        except Exception as e:                   # one retry: show the model the error and let it fix the query
            try:
                fixed, _ = llm.fix_sql(question, sql, str(e))
                if not fixed:
                    raise RuntimeError(e)
                sql, result = fixed, db.run(fixed)
            except Exception as e2:
                return {"kind": "error", "text": f"Query failed: {e2}", "sql": sql}
        try:
            text = llm.answer(question, sql, result.head(50).to_csv(index=False))
        except Exception:
            text = ""                            # the table still answers the question
        return {"text": text, "sql": sql, "result": result}

    messages = st.container()                    # history sits above the input box
    with messages:
        for msg in st.session_state.chat:
            render(msg)

    question = st.chat_input("Ask a question, e.g. Which hour of the day has the most fraud?  (or type a SELECT)",
                             disabled=not db_ready)
    if question and question.strip():
        user_msg = {"role": "user", "text": question.strip()}
        with messages:
            render(user_msg)
            with st.spinner("Thinking..."):
                reply = {"role": "assistant", **respond(question.strip())}
            render(reply)
        st.session_state.chat += [user_msg, reply]

# ---------------------------------------------------------------- predict
with predict_tab:
    with st.form("predict"):
        c1, c2 = st.columns(2)
        ttype = c1.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        hour = c2.number_input("Hour of day", 0, 23, value=None, placeholder="0–23")
        amount = c1.number_input("Amount ($)", 0.0, value=None, step=1_000.0, placeholder="e.g. 250000")
        org = c2.number_input("Sender balance before ($)", 0.0, value=None, step=1_000.0, placeholder="e.g. 250000")
        dest = c1.number_input("Receiver balance before ($)", 0.0, value=None, step=1_000.0, placeholder="e.g. 0")
        go = st.form_submit_button("Score it", type="primary")

    if go:
        if ttype not in ("TRANSFER", "CASH_OUT"):
            p = 0.0
        elif amount is None:
            st.warning("Enter an amount.")
            st.stop()
        else:
            x = pd.DataFrame([[int(ttype == "TRANSFER"), amount, org or 0.0, dest or 0.0, hour or 0]], columns=FEATURES)
            with st.spinner("Loading the model..."):
                p = float(get_model().predict_proba(x)[:, 1][0])
        st.metric("Fraud risk score", f"{p:.1%}")
        st.progress(p)
        if p >= 0.9:
            st.error("Hold for review")
        elif p >= 0.5:
            st.warning("Review")
        else:
            st.success("Approve")
