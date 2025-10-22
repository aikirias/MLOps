INSERT INTO ops.bt_crypto_events (
    site_id,
    user_id,
    purchase_date,
    crypto_type,
    purchase_price,
    purchase_units,
    purchase_value,
    is_active,
    updated_at
)
SELECT
    site_id,
    user_id,
    purchase_date,
    crypto_type,
    purchase_price,
    purchase_units,
    purchase_value,
    TRUE,
    timezone('utc', now())
FROM staging.bt_crypto_transaction_history
ON CONFLICT (site_id, user_id, purchase_date, crypto_type)
DO UPDATE SET
    purchase_price = EXCLUDED.purchase_price,
    purchase_units = EXCLUDED.purchase_units,
    purchase_value = EXCLUDED.purchase_value,
    is_active = TRUE,
    updated_at = timezone('utc', now());
