from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from ml.train import train_and_register


default_args = {
    "owner": "data-science",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="churn_train_register_m",
    schedule_interval="@monthly",
    start_date=datetime(2019, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "training"],
) as dag:
    snapshot_param = "{{ data_interval_end | ds }}"

    train_model = PythonOperator(
        task_id="train_and_register_model",
        python_callable=train_and_register,
        op_kwargs={
            "snapshot_date": snapshot_param,
            "lookback_weeks": 8,
        },
    )
