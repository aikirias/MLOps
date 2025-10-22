from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

FEATURE_SQL = "sql/features_churn.sql"


def validate_features(snapshot_date: str) -> None:
    hook = PostgresHook(postgres_conn_id="postgres_default")
    total_records = hook.get_first(
        """
        SELECT COUNT(*)
        FROM features.churn_weekly_features
        WHERE snapshot_date = %s
        """,
        parameters=(snapshot_date,),
    )
    if not total_records or total_records[0] == 0:
        raise ValueError(f"No features generated for snapshot_date={snapshot_date}")

    null_customers = hook.get_first(
        """
        SELECT COUNT(*)
        FROM features.churn_weekly_features
        WHERE snapshot_date = %s
          AND customer_id IS NULL
        """,
        parameters=(snapshot_date,),
    )
    if null_customers and null_customers[0] > 0:
        raise ValueError(f"Found {null_customers[0]} rows without customer_id for snapshot_date={snapshot_date}")


with DAG(
    dag_id="churn_feature_build_w",
    schedule_interval="@weekly",
    start_date=datetime(2019, 1, 1),
    catchup=False,
    default_args={
        "owner": "data-science",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    template_searchpath=["/opt/airflow/include"],
    tags=["churn", "features"],
) as dag:
    snapshot_param = "{{ data_interval_end | ds }}"

    build_features = PostgresOperator(
        task_id="build_weekly_features",
        postgres_conn_id="postgres_default",
        sql=FEATURE_SQL,
        params={"snapshot_date": snapshot_param},
    )

    validate = PythonOperator(
        task_id="validate_features",
        python_callable=validate_features,
        op_kwargs={"snapshot_date": snapshot_param},
    )

    build_features >> validate
