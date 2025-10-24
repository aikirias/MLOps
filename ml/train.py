from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import pandas as pd

import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, functions as F


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
    return df


def _create_spark_session() -> SparkSession:
    master = os.getenv("SPARK_MASTER_URL", "local[*]")
    app_name = os.getenv("SPARK_APP_NAME", "churn-train")
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "32"))
        .getOrCreate()
    )
    return spark


def _prepare_spark_dataframe(spark: SparkSession, df: pd.DataFrame) -> DataFrame:
    spark_df = spark.createDataFrame(df)

    # Cast numeric / binary features and label to double
    for col in NUMERIC_FEATURES + BINARY_FEATURES + ["labeled_churn"]:
        if col in spark_df.columns:
            spark_df = spark_df.withColumn(col, F.col(col).cast("double"))

    # Fill missing categorical values
    fill_values = {col: "unknown" for col in CATEGORICAL_FEATURES if col in spark_df.columns}
    if fill_values:
        spark_df = spark_df.fillna(fill_values)

    return spark_df


def train_and_register(snapshot_date: str, lookback_weeks: int = 8) -> str:
    config = TrainingConfig(snapshot_date=snapshot_date, lookback_weeks=lookback_weeks)
    df = _load_training_data(config)

    spark = _create_spark_session()
    try:
        spark_df = _prepare_spark_dataframe(spark, df)
        available_categoricals = [col for col in CATEGORICAL_FEATURES if col in spark_df.columns]

        indexers = [
            StringIndexer(inputCol=feature, outputCol=f"{feature}_idx", handleInvalid="keep")
            for feature in available_categoricals
        ]
        encoders = [
            OneHotEncoder(inputCol=f"{feature}_idx", outputCol=f"{feature}_oh", handleInvalid="keep")
            for feature in available_categoricals
        ]

        assembler_inputs: List[str] = []
        assembler_inputs.extend([col for col in NUMERIC_FEATURES if col in spark_df.columns])
        assembler_inputs.extend([col for col in BINARY_FEATURES if col in spark_df.columns])
        assembler_inputs.extend([f"{feature}_oh" for feature in available_categoricals])
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

        lr = LogisticRegression(
            featuresCol="features",
            labelCol="labeled_churn",
            maxIter=50,
            elasticNetParam=0.0,
        )

        pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])
        train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
        logger.info(
            "Prepared dataset: train=%d rows, test=%d rows",
            train_df.count(),
            test_df.count(),
        )

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("churn-logreg-experiment")

        with mlflow.start_run(run_name=f"train_{snapshot_date}") as run:
            model = pipeline.fit(train_df)
            predictions = model.transform(test_df)

            evaluator = BinaryClassificationEvaluator(
                labelCol="labeled_churn", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
            )
            roc_auc = evaluator.evaluate(predictions)
            accuracy = predictions.select(
                F.mean((F.col("prediction") == F.col("labeled_churn")).cast("double")).alias("acc")
            ).collect()[0]["acc"]

            positive_rate_train = train_df.agg(F.mean("labeled_churn").alias("rate")).collect()[0]["rate"]
            positive_rate_test = test_df.agg(F.mean("labeled_churn").alias("rate")).collect()[0]["rate"]

            metrics = {
                "roc_auc": float(roc_auc),
                "accuracy": float(accuracy),
                "positive_rate_train": float(positive_rate_train),
                "positive_rate_test": float(positive_rate_test),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_params(
                {
                    "snapshot_date": snapshot_date,
                    "lookback_weeks": lookback_weeks,
                    "numeric_features": ",".join(assembler_inputs),
                    "categorical_features": ",".join(available_categoricals),
                }
            )
            logger.info(
                "Evaluation metrics | accuracy=%.4f roc_auc=%.4f pos_rate_train=%.4f pos_rate_test=%.4f",
                metrics["accuracy"],
                metrics["roc_auc"],
                metrics["positive_rate_train"],
                metrics["positive_rate_test"],
            )

            model_name = get_model_name()
            mlflow.spark.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name,
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
    finally:
        spark.stop()

    return version


if __name__ == "__main__":
    snapshot = os.getenv("SNAPSHOT_DATE", datetime.utcnow().strftime("%Y-%m-%d"))
    train_and_register(snapshot)
