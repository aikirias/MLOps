WITH params AS (
    SELECT DATE '{{ data_interval_end | ds }}' AS snapshot_date
),
customer_base AS (
    SELECT DISTINCT cus_cust_id_buy AS customer_id FROM raw.payments
    UNION
    SELECT DISTINCT cus_cust_id_buy FROM raw.marketplace_data
    UNION
    SELECT DISTINCT cus_cust_id_buy FROM raw.active_user
    UNION
    SELECT DISTINCT cus_cust_id_buy FROM raw.demograficos
    UNION
    SELECT DISTINCT CAST(cus_cust_id_buy AS BIGINT) FROM raw.dinero_cuenta
),
payment_metrics AS (
    SELECT
        p.cus_cust_id_buy AS customer_id,
        SUM(p.spent) FILTER (WHERE p.fecha >= params.snapshot_date - INTERVAL '7 day' AND p.fecha < params.snapshot_date) AS spent_7d,
        SUM(p.spent) FILTER (WHERE p.fecha >= params.snapshot_date - INTERVAL '30 day' AND p.fecha < params.snapshot_date) AS spent_30d,
        SUM(p.spent) FILTER (WHERE p.fecha >= params.snapshot_date - INTERVAL '90 day' AND p.fecha < params.snapshot_date) AS spent_90d,
        COUNT(*) FILTER (WHERE p.fecha >= params.snapshot_date - INTERVAL '30 day' AND p.fecha < params.snapshot_date) AS payments_30d,
        COUNT(*) FILTER (WHERE p.fecha >= params.snapshot_date - INTERVAL '90 day' AND p.fecha < params.snapshot_date) AS payments_90d,
        SUM(p.descuento) FILTER (WHERE p.fecha >= params.snapshot_date - INTERVAL '30 day' AND p.fecha < params.snapshot_date) AS discount_30d,
        SUM(p.descuento) FILTER (WHERE p.fecha >= params.snapshot_date - INTERVAL '90 day' AND p.fecha < params.snapshot_date) AS discount_90d,
        MAX(p.fecha) AS last_payment_date
    FROM raw.payments p
    CROSS JOIN params
    WHERE p.fecha < params.snapshot_date
    GROUP BY p.cus_cust_id_buy
),
activity_metrics AS (
    SELECT
        a.cus_cust_id_buy AS customer_id,
        MAX(a.mau_mp_1) AS mau_mp_1,
        MAX(a.mau_ml_1) AS mau_ml_1,
        MAX(a.mau_mp_3) AS mau_mp_3,
        MAX(a.mau_ml_3) AS mau_ml_3,
        MAX(a.last_login_mp_date_1) AS last_login_mp_date_1,
        MAX(a.last_login_ml_date_1) AS last_login_ml_date_1,
        GREATEST(
            COALESCE(MAX(a.last_login_mp_date_1), DATE '1970-01-01'),
            COALESCE(MAX(a.last_login_ml_date_1), DATE '1970-01-01')
        ) AS last_login_date
    FROM raw.active_user a
    GROUP BY a.cus_cust_id_buy
),
dinero_metrics AS (
    SELECT
        CAST(d.cus_cust_id_buy AS BIGINT) AS customer_id,
        MAX(d.plata_cuenta_1) AS plata_cuenta_1,
        MAX(d.plata_cuenta_2) AS plata_cuenta_2,
        MAX(CASE WHEN LOWER(d.inversion) IN ('eligible', 'active', 'invested') THEN 1 ELSE 0 END) = 1 AS has_inversion
    FROM raw.dinero_cuenta d
    GROUP BY CAST(d.cus_cust_id_buy AS BIGINT)
),
demograficos AS (
    SELECT
        g.cus_cust_id_buy AS customer_id,
        MAX(g.gender) AS gender,
        MAX(g.rango_edad) AS rango_edad,
        MAX(g.tarjetas) AS tarjetas,
        MAX(g.estado) AS estado
    FROM raw.demograficos g
    GROUP BY g.cus_cust_id_buy
),
marketplace AS (
    SELECT
        m.cus_cust_id_buy AS customer_id,
        MAX(m.spent_ml) AS spent_ml,
        MAX(m.recency_ml) AS recency_ml,
        MAX(m.frequency_ml) AS frequency_ml
    FROM raw.marketplace_data m
    GROUP BY m.cus_cust_id_buy
),
assembled AS (
    SELECT
        b.customer_id,
        params.snapshot_date,
        COALESCE(p.spent_7d, 0) AS spent_7d,
        COALESCE(p.spent_30d, 0) AS spent_30d,
        COALESCE(p.spent_90d, 0) AS spent_90d,
        COALESCE(p.payments_30d, 0) AS payments_30d,
        COALESCE(p.payments_90d, 0) AS payments_90d,
        COALESCE(p.discount_30d, 0) AS total_discount_30d,
        COALESCE(a.mau_mp_1, 0) AS mau_mp_1,
        COALESCE(a.mau_ml_1, 0) AS mau_ml_1,
        COALESCE(a.mau_mp_3, 0) AS mau_mp_3,
        COALESCE(a.mau_ml_3, 0) AS mau_ml_3,
        COALESCE(d.plata_cuenta_1, 0) AS plata_cuenta_1,
        COALESCE(d.plata_cuenta_2, 0) AS plata_cuenta_2,
        COALESCE(d.has_inversion, FALSE) AS has_inversion,
        demo.gender,
        demo.rango_edad,
        demo.tarjetas,
        demo.estado,
        p.last_payment_date,
        a.last_login_date,
        GREATEST(
            COALESCE(p.last_payment_date, DATE '1970-01-01'),
            COALESCE(a.last_login_date, DATE '1970-01-01')
        ) AS last_activity_date,
        m.spent_ml,
        m.recency_ml,
        m.frequency_ml
    FROM customer_base b
    CROSS JOIN params
    LEFT JOIN payment_metrics p ON p.customer_id = b.customer_id
    LEFT JOIN activity_metrics a ON a.customer_id = b.customer_id
    LEFT JOIN dinero_metrics d ON d.customer_id = b.customer_id
    LEFT JOIN demograficos demo ON demo.customer_id = b.customer_id
    LEFT JOIN marketplace m ON m.customer_id = b.customer_id
),
labeled AS (
    SELECT
        a.customer_id,
        a.snapshot_date,
        a.spent_7d,
        a.spent_30d,
        a.spent_90d,
        a.payments_30d,
        a.payments_90d,
        a.total_discount_30d,
        a.mau_mp_1,
        a.mau_ml_1,
        a.mau_mp_3,
        a.mau_ml_3,
        a.plata_cuenta_1,
        a.plata_cuenta_2,
        a.has_inversion,
        a.gender,
        a.rango_edad,
        a.tarjetas,
        a.estado,
        a.last_payment_date,
        a.last_login_date,
        a.last_activity_date,
        a.spent_ml,
        a.recency_ml,
        a.frequency_ml,
        (a.snapshot_date - COALESCE(a.last_activity_date, a.snapshot_date))::INTEGER AS days_since_activity,
        COALESCE(e.churn,
            CASE
                WHEN a.last_activity_date IS NULL THEN 1
                WHEN (a.snapshot_date - a.last_activity_date) >= 60 THEN 1
                ELSE 0
            END
        ) AS labeled_churn
    FROM assembled a
    LEFT JOIN raw.evaluate e ON e.cus_cust_id_buy = a.customer_id
)
INSERT INTO features.churn_weekly_features (
    customer_id,
    snapshot_date,
    spent_7d,
    spent_30d,
    payments_30d,
    total_discount_30d,
    spent_90d,
    payments_90d,
    mau_mp_1,
    mau_ml_1,
    mau_mp_3,
    mau_ml_3,
    plata_cuenta_1,
    plata_cuenta_2,
    has_inversion,
    gender,
    rango_edad,
    tarjetas,
    estado,
    days_since_activity,
    last_payment_date,
    last_login_date,
    labeled_churn
)
SELECT
    customer_id,
    snapshot_date,
    spent_7d,
    spent_30d,
    payments_30d,
    total_discount_30d,
    spent_90d,
    payments_90d,
    mau_mp_1,
    mau_ml_1,
    mau_mp_3,
    mau_ml_3,
    plata_cuenta_1,
    plata_cuenta_2,
    has_inversion,
    gender,
    rango_edad,
    tarjetas,
    estado,
    days_since_activity,
    last_payment_date,
    last_login_date,
    labeled_churn
FROM labeled
ON CONFLICT (customer_id, snapshot_date)
DO UPDATE SET
    spent_7d = EXCLUDED.spent_7d,
    spent_30d = EXCLUDED.spent_30d,
    payments_30d = EXCLUDED.payments_30d,
    total_discount_30d = EXCLUDED.total_discount_30d,
    spent_90d = EXCLUDED.spent_90d,
    payments_90d = EXCLUDED.payments_90d,
    mau_mp_1 = EXCLUDED.mau_mp_1,
    mau_ml_1 = EXCLUDED.mau_ml_1,
    mau_mp_3 = EXCLUDED.mau_mp_3,
    mau_ml_3 = EXCLUDED.mau_ml_3,
    plata_cuenta_1 = EXCLUDED.plata_cuenta_1,
    plata_cuenta_2 = EXCLUDED.plata_cuenta_2,
    has_inversion = EXCLUDED.has_inversion,
    gender = EXCLUDED.gender,
    rango_edad = EXCLUDED.rango_edad,
    tarjetas = EXCLUDED.tarjetas,
    estado = EXCLUDED.estado,
    days_since_activity = EXCLUDED.days_since_activity,
    last_payment_date = EXCLUDED.last_payment_date,
    last_login_date = EXCLUDED.last_login_date,
    labeled_churn = EXCLUDED.labeled_churn,
    created_at = timezone('utc', now());
