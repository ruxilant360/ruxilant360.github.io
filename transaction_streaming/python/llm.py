r"""Turn a plain-English question into one PostgreSQL SELECT query, and answer it in a sentence, using a
small free model on Groq (https://console.groq.com).

Setup: put your key in transaction_streaming/python/.env (copy .env.example):
    GROQ_API_KEY=gsk_...
Optional: GROQ_MODEL=openai/gpt-oss-20b (the default). See which models your key can use at
https://console.groq.com/docs/models
"""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM = """You write PostgreSQL queries for a fraud analytics app.

Database: one table, `transactions`, a 20,000-row sample of simulated mobile-money transactions.
All column names are lower-case; never quote them.
Columns:
- transaction_id TEXT          unique id, e.g. 'tx000123'
- step INTEGER                 hour since the simulation started (1-743, about 31 days). Hour of day = step % 24, day = step / 24 + 1
- type TEXT                    one of 'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'
- amount DOUBLE PRECISION      transaction amount
- nameorig TEXT                sender account id
- oldbalanceorg DOUBLE PRECISION   sender balance before the transaction
- newbalanceorig DOUBLE PRECISION  sender balance after
- namedest TEXT                receiver account id ('M...' ids are merchants)
- oldbalancedest DOUBLE PRECISION  receiver balance before
- newbalancedest DOUBLE PRECISION  receiver balance after
- isflaggedfraud INTEGER       1 if the bank's existing rule flagged it
- isfraud INTEGER              1 if the transaction was fraud

Note: the sample keeps every fraud row (1,348) and a random sample of normal rows, so fraud
makes up about 6.7% of the sample, far more than the 0.13% in the full data. When a question
asks for a fraud rate, compute it, and the app will show that caveat.

Rules:
- Answer with exactly one read-only SELECT (or WITH ... SELECT) statement, inside a ```sql block.
- Use PostgreSQL syntax. ROUND(x, n) needs a numeric argument: write ROUND(x::numeric, n).
  Integer division truncates: use AVG(isfraud) or 100.0 * SUM(isfraud) / COUNT(*) for rates.
- Use only the table and columns above. Give computed columns short readable aliases.
- Add LIMIT 200 unless the query already aggregates to a small result.
- If the question can't be answered from this table, reply with one sentence saying why and no SQL."""

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq                   # imported here so the page loads before the SDK
        _client = Groq(api_key=os.getenv("GROQ_API_KEY", "").strip().strip('"'), timeout=30, max_retries=2)
    return _client


def is_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip().strip('"'))


def _chat(system: str, messages: list[dict], max_tokens: int = 2000) -> str:
    kwargs = {}
    if MODEL.startswith("openai/gpt-oss"):      # reasoning models: keep the thinking short and out of the reply
        kwargs = {"reasoning_effort": "low", "include_reasoning": False}
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, *messages],
        temperature=0,
        max_completion_tokens=max_tokens,
        **kwargs,
    )
    text = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # some models (qwen) inline their thinking


def _ask(messages: list[dict]) -> tuple[str | None, str]:
    text = _chat(SYSTEM, messages)
    match = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    if match:
        return match.group(1).strip(), text
    if re.match(r"(?is)^\s*(select|with)\b", text):    # SQL without a code fence
        return text.strip(), text
    return None, text or "The model returned an empty reply. Try rephrasing the question."


def question_to_sql(question: str) -> tuple[str | None, str]:
    """Return (sql, raw_text). sql is None when the model says the question can't be answered."""
    return _ask([{"role": "user", "content": question}])


def fix_sql(question: str, sql: str, error: str) -> tuple[str | None, str]:
    """Give the model the PostgreSQL error and ask for a corrected query."""
    return _ask([
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"```sql\n{sql}\n```"},
        {"role": "user", "content": f"PostgreSQL returned this error:\n{error}\n\nReply with a corrected query."},
    ])


ANSWER_SYSTEM = """You answer questions about a 20,000-row sample of simulated mobile-money transactions.
You get the question, the SQL that was run and its result as CSV. Reply in one or two plain sentences that answer
the question with the key numbers from the result. Don't describe the SQL. Don't invent numbers that aren't in the
result. If the result is empty, say so. The sample keeps every fraud row, so fraud rates in it are far higher than in
the real data (about 6.7% vs 0.13% overall); if you quote a fraud rate, add "in this sample"."""


def answer(question: str, sql: str, result_csv: str) -> str:
    """A short plain-English answer from the query result."""
    return _chat(ANSWER_SYSTEM, [{"role": "user", "content":
                 f"Question: {question}\n\nSQL:\n{sql}\n\nResult (CSV, up to 50 rows):\n{result_csv}"}],
                 max_tokens=1000)
