from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from ml.infer_batch import score_snapshot


default_args = {
    "owner": "data-science",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="churn_batch_score_w",
    schedule_interval="@weekly",
    start_date=datetime(2019, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "inference"],
) as dag:
    snapshot_param = "{{ data_interval_end | ds }}"

    batch_score = PythonOperator(
        task_id="score_batch",
        python_callable=score_snapshot,
        op_kwargs={"snapshot_date": snapshot_param},
    )
