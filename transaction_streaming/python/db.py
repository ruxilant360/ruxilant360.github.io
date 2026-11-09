"""PostgreSQL access for the Streamlit app.

The 20,000-row sample is loaded into a `fraud.transactions` table the first time the app runs
(skipped if the table already has rows). Questions are answered with read-only queries.

Connection settings come from transaction_streaming/python/.env:
    DATABASE_URL=postgresql://postgres:<password>@localhost:5432/postgres
"""
import os
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"
SCHEMA_NAME = "fraud"        # own schema, so it never clashes with a `transactions` table from another project
TABLE = "transactions"

# Column names are stored lower-case so queries don't need "quoted" identifiers.
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{TABLE} (
    transaction_id  TEXT PRIMARY KEY,
    step            INTEGER,
    type            TEXT,
    amount          DOUBLE PRECISION,
    nameorig        TEXT,
    oldbalanceorg   DOUBLE PRECISION,
    newbalanceorig  DOUBLE PRECISION,
    namedest        TEXT,
    oldbalancedest  DOUBLE PRECISION,
    newbalancedest  DOUBLE PRECISION,
    isflaggedfraud  INTEGER,
    isfraud         INTEGER
)"""


def database_url() -> str | None:
    return os.getenv("DATABASE_URL")


def connect() -> psycopg.Connection:
    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set in transaction_streaming/python/.env")
    con = psycopg.connect(url, connect_timeout=5)
    con.execute(f"SET search_path TO {SCHEMA_NAME}")   # plain `transactions` in queries means fraud.transactions
    return con


def setup() -> int:
    """Create the table and load the sample if it is empty. Returns the row count."""
    with connect() as con:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
        con.execute(SCHEMA)
        (count,) = con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()
        if count == 0:
            df = pd.read_csv(SAMPLE)
            df.columns = [c.lower() for c in df.columns]
            with con.cursor() as cur, cur.copy(f"COPY {TABLE} ({', '.join(df.columns)}) FROM STDIN") as copy:
                for row in df.itertuples(index=False):
                    copy.write_row(row)
            count = len(df)
        con.commit()
    return count


def run(sql: str, max_rows: int = 5000) -> pd.DataFrame:
    """Run one SELECT inside a READ ONLY transaction, so nothing can be changed."""
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("Run one statement at a time.")
    with connect() as con:
        con.execute("SET TRANSACTION READ ONLY")
        con.execute("SET LOCAL statement_timeout = '15s'")
        with con.cursor() as cur:
            cur.execute(sql)
            if cur.description is None:
                raise ValueError("The query returned no table.")
            cols = [d.name for d in cur.description]
            rows = cur.fetchmany(max_rows)
        con.rollback()
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:                      # NUMERIC results (avg, sum, round) arrive as Decimal
        if df[c].map(lambda v: isinstance(v, Decimal) or v is None).all() and df[c].notna().any():
            df[c] = df[c].astype(float)
    return df
