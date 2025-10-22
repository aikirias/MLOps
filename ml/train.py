from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow


logger = logging.getLogger(__name__)

try:
    from .model_config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES  # type: ignore
    from .utils_io import fetch_dataframe, get_engine, get_mlflow_client, get_model_name  # type: ignore
except ImportError:  # pragma: no cover - fallback for script execution
    from model_config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
    from utils_io import fetch_dataframe, get_engine, get_mlflow_client, get_model_name


@dataclass
class TrainingConfig:
    snapshot_date: str
    lookback_weeks: int = 8

    @property
    def snapshot_dt(self) -> datetime:
        return datetime.strptime(self.snapshot_date, "%Y-%m-%d")

    @property
    def window_start(self) -> datetime:
        return self.snapshot_dt - timedelta(weeks=self.lookback_weeks)


def _load_training_data(config: TrainingConfig) -> pd.DataFrame:
    query = """
        SELECT *
        FROM features.churn_weekly_features
        WHERE snapshot_date BETWEEN :window_start AND :snapshot_date
    """
    engine = get_engine()
    df = fetch_dataframe(
        query,
        params={
            "window_start": config.window_start.date(),
            "snapshot_date": config.snapshot_dt.date(),
        },
        engine=engine,
    )
    df = df.drop_duplicates(subset=["customer_id", "snapshot_date"])
    df = df.dropna(subset=["labeled_churn"])
    logger.info(
        "Loaded training window %s – %s with %d rows",
        config.window_start.date(),
        config.snapshot_dt.date(),
        len(df),
    )
    numeric_like = [col for col in NUMERIC_FEATURES if col in df.columns]
    for col in numeric_like:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        raise ValueError("No training data available for the requested window")
    return df


def _split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    available_numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    available_binary = [col for col in BINARY_FEATURES if col in df.columns]
    available_categoricals = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    X = df[available_numeric + available_binary + available_categoricals].copy()
    y = df["labeled_churn"].astype(int)

    for binary_col in available_binary:
        X[binary_col] = X[binary_col].astype(int)

    numeric_cols = available_numeric + available_binary
    return X, y, numeric_cols, available_categoricals


def _build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_and_register(snapshot_date: str, lookback_weeks: int = 8) -> str:
    config = TrainingConfig(snapshot_date=snapshot_date, lookback_weeks=lookback_weeks)
    df = _load_training_data(config)
    X, y, numeric_cols, categorical_cols = _split_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        "Prepared dataset: train=%d rows, test=%d rows, features_num=%d, features_cat=%d",
        len(X_train),
        len(X_test),
        len(numeric_cols),
        len(categorical_cols),
    )

    pipeline = _build_pipeline(numeric_cols, categorical_cols)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("churn-logreg-experiment")

    with mlflow.start_run(run_name=f"train_{snapshot_date}") as run:
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "positive_rate_train": float(np.mean(y_train)),
            "positive_rate_test": float(np.mean(y_test)),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "snapshot_date": snapshot_date,
                "lookback_weeks": lookback_weeks,
                "numeric_features": ",".join(numeric_cols),
                "categorical_features": ",".join(categorical_cols),
            }
        )
        logger.info(
            "Evaluation metrics | accuracy=%.4f roc_auc=%.4f pos_rate_train=%.4f pos_rate_test=%.4f",
            metrics["accuracy"],
            metrics["roc_auc"],
            metrics["positive_rate_train"],
            metrics["positive_rate_test"],
        )

        signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
        model_name = get_model_name()
        model_info = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=X_train.head(1),
        )

        client = get_mlflow_client()
        registered_versions = client.search_model_versions(
            filter_string=f"run_id='{run.info.run_id}'"
        )
        if not registered_versions:
            raise RuntimeError("Model registration failed")

        version = registered_versions[0].version
        logger.info("Registered model %s version %s", model_name, version)
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Promoted model %s version %s to Production", model_name, version)

        mlflow.set_tag("model_version", version)
        mlflow.set_tag("production_ready", True)

    return version


if __name__ == "__main__":
    snapshot = os.getenv("SNAPSHOT_DATE", datetime.utcnow().strftime("%Y-%m-%d"))
    train_and_register(snapshot)
