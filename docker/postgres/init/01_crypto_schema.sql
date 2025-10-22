CREATE TABLE IF NOT EXISTS raw.bt_crypto_transaction_history (
    site_id TEXT,
    user_id BIGINT,
    purchase_date TEXT,
    crypto_type TEXT,
    purchase_price NUMERIC,
    purchase_units NUMERIC
);

CREATE TABLE IF NOT EXISTS staging.bt_crypto_transaction_history (
    site_id TEXT NOT NULL,
    user_id BIGINT NOT NULL,
    purchase_date DATE NOT NULL,
    crypto_type TEXT NOT NULL,
    purchase_price NUMERIC(18, 4) NOT NULL,
    purchase_units NUMERIC(18, 8) NOT NULL,
    purchase_value NUMERIC(18, 8) NOT NULL,
    load_ts TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

CREATE TABLE IF NOT EXISTS ops.bt_crypto_events (
    site_id TEXT NOT NULL,
    user_id BIGINT NOT NULL,
    purchase_date DATE NOT NULL,
    crypto_type TEXT NOT NULL,
    purchase_price NUMERIC(18, 4) NOT NULL,
    purchase_units NUMERIC(18, 8) NOT NULL,
    purchase_value NUMERIC(18, 8) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    PRIMARY KEY (site_id, user_id, purchase_date, crypto_type)
);

CREATE INDEX IF NOT EXISTS idx_bt_crypto_events_user ON ops.bt_crypto_events (user_id);
