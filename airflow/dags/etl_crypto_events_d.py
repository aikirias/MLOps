from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

LOAD_STAGING_SQL = """
TRUNCATE TABLE staging.bt_crypto_transaction_history;
WITH cleaned AS (
    SELECT
        UPPER(TRIM(site_id)) AS site_id,
        user_id,
        TO_DATE(purchase_date, 'DD/MM/YYYY') AS purchase_date,
        UPPER(TRIM(crypto_type)) AS crypto_type,
        CAST(purchase_price AS NUMERIC(18, 4)) AS purchase_price,
        CAST(purchase_units AS NUMERIC(18, 8)) AS purchase_units,
        CAST(purchase_price AS NUMERIC(18, 4)) * CAST(purchase_units AS NUMERIC(18, 8)) AS purchase_value,
        ROW_NUMBER() OVER (
            PARTITION BY UPPER(TRIM(site_id)), user_id, TO_DATE(purchase_date, 'DD/MM/YYYY'), UPPER(TRIM(crypto_type))
            ORDER BY TO_DATE(purchase_date, 'DD/MM/YYYY') DESC, CAST(purchase_price AS NUMERIC(18, 4)) DESC
        ) AS row_num
    FROM raw.bt_crypto_transaction_history
    WHERE site_id IS NOT NULL
      AND user_id IS NOT NULL
      AND purchase_date IS NOT NULL
      AND crypto_type IS NOT NULL
      AND purchase_price IS NOT NULL
      AND purchase_units IS NOT NULL
),
deduped AS (
    SELECT *
    FROM cleaned
    WHERE row_num = 1
)
INSERT INTO staging.bt_crypto_transaction_history (
    site_id,
    user_id,
    purchase_date,
    crypto_type,
    purchase_price,
    purchase_units,
    purchase_value
)
SELECT
    site_id,
    user_id,
    purchase_date,
    crypto_type,
    purchase_price,
    purchase_units,
    purchase_value
FROM deduped;
"""

DEACTIVATE_MISSING_SQL = """
UPDATE ops.bt_crypto_events events
SET is_active = FALSE,
    updated_at = timezone('utc', now())
WHERE events.purchase_date >= DATE '{{ ds }}' - INTERVAL '5 days'
  AND NOT EXISTS (
      SELECT 1
      FROM staging.bt_crypto_transaction_history staging
      WHERE staging.site_id = events.site_id
        AND staging.user_id = events.user_id
        AND staging.purchase_date = events.purchase_date
        AND staging.crypto_type = events.crypto_type
  );
"""


def validate_staging() -> None:
    hook = PostgresHook(postgres_conn_id="postgres_default")
    checks = {
        "null_required": """
            SELECT COUNT(*)
            FROM staging.bt_crypto_transaction_history
            WHERE site_id IS NULL
               OR user_id IS NULL
               OR purchase_date IS NULL
               OR crypto_type IS NULL
               OR purchase_price IS NULL
               OR purchase_units IS NULL
        """,
        "date_in_future": """
            SELECT COUNT(*)
            FROM staging.bt_crypto_transaction_history
            WHERE purchase_date > CURRENT_DATE + INTERVAL '1 day'
        """,
        "mismatched_value": """
            SELECT COUNT(*)
            FROM staging.bt_crypto_transaction_history
            WHERE ABS(purchase_value - purchase_price * purchase_units) > 0.0001
        """,
    }

    for check_name, sql in checks.items():
        records = hook.get_first(sql)
        if records and records[0] and records[0] > 0:
            raise ValueError(f"Integrity check {check_name} failed with {records[0]} offending rows")


with DAG(
    dag_id="etl_crypto_events_d",
    schedule_interval="@daily",
    start_date=datetime(2019, 1, 1),
    catchup=False,
    default_args={
        "owner": "data-platform",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    template_searchpath=["/opt/airflow/include"],
    tags=["crypto", "etl"],
) as dag:
    load_staging = PostgresOperator(
        task_id="load_staging",
        postgres_conn_id="postgres_default",
        sql=LOAD_STAGING_SQL,
    )

    validate = PythonOperator(
        task_id="validate_staging",
        python_callable=validate_staging,
    )

    upsert_events = PostgresOperator(
        task_id="upsert_crypto_events",
        postgres_conn_id="postgres_default",
        sql="sql/upsert_crypto_events.sql",
    )

    deactivate_missing = PostgresOperator(
        task_id="deactivate_missing",
        postgres_conn_id="postgres_default",
        sql=DEACTIVATE_MISSING_SQL,
    )

    load_staging >> validate >> upsert_events >> deactivate_missing
