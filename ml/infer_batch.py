from __future__ import annotations

import os
from datetime import datetime
from typing import Dict

import mlflow
import pandas as pd
from sqlalchemy import text

try:
    from .model_config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES  # type: ignore
    from .utils_io import fetch_dataframe, get_engine, get_mlflow_client, get_model_name, write_dataframe  # type: ignore
except ImportError:  # pragma: no cover - fallback for script execution
    from model_config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
    from utils_io import fetch_dataframe, get_engine, get_mlflow_client, get_model_name, write_dataframe


def _load_snapshot(snapshot_date: str) -> pd.DataFrame:
    query = """
        SELECT *
        FROM features.churn_weekly_features
        WHERE snapshot_date = :snapshot_date
    """
    df = fetch_dataframe(query, {"snapshot_date": snapshot_date})
    if df.empty:
        raise ValueError(f"No features available for snapshot_date={snapshot_date}")
    return df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    for col in NUMERIC_FEATURES:
        if col not in prepared.columns:
            prepared[col] = 0.0
        prepared[col] = prepared[col].fillna(0.0)

    for col in BINARY_FEATURES:
        if col not in prepared.columns:
            prepared[col] = 0
        prepared[col] = prepared[col].fillna(0).astype(int)

    for col in CATEGORICAL_FEATURES:
        if col not in prepared.columns:
            prepared[col] = "unknown"
        prepared[col] = prepared[col].fillna("unknown").astype(str)

    return prepared


def _load_production_model() -> Dict[str, object]:
    model_name = get_model_name()
    client = get_mlflow_client()
    latest = client.get_latest_versions(model_name, stages=["Production"])
    if not latest:
        raise RuntimeError(f"No production model found for {model_name}")

    version = latest[0].version
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    sklearn_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    return {"model": sklearn_model, "version": version}


def score_snapshot(snapshot_date: str, decision_threshold: float = 0.5) -> None:
    raw_df = _load_snapshot(snapshot_date)
    features_df = _prepare_features(raw_df)

    feature_columns = [
        *(col for col in NUMERIC_FEATURES if col in features_df.columns),
        *(col for col in BINARY_FEATURES if col in features_df.columns),
        *(col for col in CATEGORICAL_FEATURES if col in features_df.columns),
    ]

    bundle = _load_production_model()
    model = bundle["model"]
    version = bundle["version"]

    probabilities = model.predict_proba(features_df[feature_columns])[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)

    output = pd.DataFrame(
        {
            "customer_id": raw_df["customer_id"],
            "snapshot_date": pd.to_datetime(raw_df["snapshot_date"]).dt.date,
            "score": probabilities,
            "prediction": predictions,
            "model_version": str(version),
        }
    )

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM ops.churn_scoring
                WHERE snapshot_date = :snapshot_date
                  AND model_version = :model_version
                """
            ),
            {
                "snapshot_date": snapshot_date,
                "model_version": str(version),
            },
        )

    write_dataframe(output, table="churn_scoring", schema="ops", engine=engine)


if __name__ == "__main__":
    snapshot = os.getenv("SNAPSHOT_DATE", datetime.utcnow().strftime("%Y-%m-%d"))
    score_snapshot(snapshot)
