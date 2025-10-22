"""Utility helpers for database and MLflow interactions."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
import mlflow
from mlflow.tracking import MlflowClient


def _get_db_uri() -> str:
    user = os.getenv("POSTGRES_USER", "mlops")
    password = os.getenv("POSTGRES_PASSWORD", "mlops")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "mlops")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def get_engine(echo: bool = False) -> Engine:
    """Return a SQLAlchemy engine pointing to the project warehouse."""
    return create_engine(_get_db_uri(), echo=echo)


@contextmanager
def db_connection(engine: Optional[Engine] = None) -> Iterable[Connection]:
    eng = engine or get_engine()
    with eng.connect() as conn:
        yield conn


def fetch_dataframe(query: str, params: Optional[Dict[str, object]] = None, engine: Optional[Engine] = None) -> pd.DataFrame:
    eng = engine or get_engine()
    with eng.connect() as conn:
        result = conn.execute(text(query), params or {})
        rows = result.fetchall()
        columns = result.keys()
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns)


def write_dataframe(
    df: pd.DataFrame,
    table: str,
    schema: str,
    engine: Optional[Engine] = None,
    if_exists: str = "append",
) -> None:
    eng = engine or get_engine()
    if df.empty:
        return

    if if_exists == "replace":
        truncate_table(schema=schema, table=table, engine=eng)

    records = df.to_dict(orient="records")
    columns = df.columns.tolist()
    column_list = ", ".join(columns)
    value_placeholders = ", ".join(f":{col}" for col in columns)
    statement = text(
        f"INSERT INTO {schema}.{table} ({column_list}) VALUES ({value_placeholders})"
    )

    with eng.begin() as connection:
        connection.execute(statement, records)


def truncate_table(schema: str, table: str, engine: Optional[Engine] = None) -> None:
    eng = engine or get_engine()
    with eng.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {schema}.{table}"))


def get_mlflow_client() -> MlflowClient:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def get_model_name() -> str:
    return os.getenv("MLFLOW_MODEL_NAME", "churn-logreg")
