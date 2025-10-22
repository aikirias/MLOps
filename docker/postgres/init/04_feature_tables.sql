CREATE TABLE IF NOT EXISTS features.churn_weekly_features (
    customer_id BIGINT NOT NULL,
    snapshot_date DATE NOT NULL,
    spent_7d NUMERIC,
    spent_30d NUMERIC,
    payments_30d INTEGER,
    total_discount_30d NUMERIC,
    spent_90d NUMERIC,
    payments_90d INTEGER,
    mau_mp_1 NUMERIC,
    mau_ml_1 NUMERIC,
    mau_mp_3 NUMERIC,
    mau_ml_3 NUMERIC,
    plata_cuenta_1 NUMERIC,
    plata_cuenta_2 NUMERIC,
    has_inversion BOOLEAN,
    gender TEXT,
    rango_edad TEXT,
    tarjetas TEXT,
    estado TEXT,
    days_since_activity INTEGER,
    last_payment_date DATE,
    last_login_date DATE,
    labeled_churn NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    PRIMARY KEY (customer_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS ops.churn_scoring (
    customer_id BIGINT NOT NULL,
    snapshot_date DATE NOT NULL,
    score NUMERIC,
    prediction INTEGER,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    PRIMARY KEY (customer_id, snapshot_date, model_version)
);
