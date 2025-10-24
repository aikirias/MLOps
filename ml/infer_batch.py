from __future__ import annotations

import os
from datetime import datetime
from typing import Dict

import mlflow
import mlflow.artifacts
import pandas as pd
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SparkSession
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


def _create_spark_session(app_name: str = "churn-infer") -> SparkSession:
    master = os.getenv("SPARK_MASTER_URL", "local[*]")
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "32"))
        .getOrCreate()
    )

    hadoop_conf = spark._jsc.hadoopConfiguration()
    endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    hadoop_conf.set("fs.s3a.endpoint", endpoint)
    hadoop_conf.set("fs.s3a.access.key", access_key)
    hadoop_conf.set("fs.s3a.secret.key", secret_key)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.fast.upload", "true")
    hadoop_conf.set(
        "fs.s3a.connection.ssl.enabled", str(endpoint.startswith("https")).lower()
    )
    hadoop_conf.set(
        "fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
    )
    return spark


def _load_production_model(spark: SparkSession) -> Dict[str, object]:
    model_name = get_model_name()
    client = get_mlflow_client()
    latest = client.get_latest_versions(model_name, stages=["Production"])
    if not latest:
        raise RuntimeError(f"No production model found for {model_name}")

    version = latest[0].version
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    local_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}/Production"
    )
    model_path = os.path.join(local_dir, "sparkml")
    spark_model = PipelineModel.load(model_path)
    return {"model": spark_model, "version": version}


def score_snapshot(snapshot_date: str, decision_threshold: float = 0.5) -> None:
    raw_df = _load_snapshot(snapshot_date)
    features_df = _prepare_features(raw_df)

    relevant_columns = [
        *(col for col in NUMERIC_FEATURES if col in features_df.columns),
        *(col for col in BINARY_FEATURES if col in features_df.columns),
        *(col for col in CATEGORICAL_FEATURES if col in features_df.columns),
        "customer_id",
        "snapshot_date",
    ]

    spark = _create_spark_session()
    try:
        bundle = _load_production_model(spark)
        model = bundle["model"]
        version = bundle["version"]

        spark_features = spark.createDataFrame(features_df[relevant_columns])
        predictions_spark = model.transform(spark_features)
        predictions_pdf = predictions_spark.select(
            "customer_id", "snapshot_date", "probability", "prediction"
        ).toPandas()

        probabilities = predictions_pdf["probability"].apply(
            lambda v: float(v[1]) if hasattr(v, "__getitem__") else float(v)
        )
        predicted_labels = (probabilities >= decision_threshold).astype(int)

        output = pd.DataFrame(
            {
                "customer_id": predictions_pdf["customer_id"],
                "snapshot_date": pd.to_datetime(predictions_pdf["snapshot_date"]).dt.date,
                "score": probabilities.values,
                "prediction": predicted_labels.values,
                "model_version": str(version),
            }
        )
    finally:
        spark.stop()

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
