"""Shared model configuration for churn workflows."""

NUMERIC_FEATURES = [
    "spent_7d",
    "spent_30d",
    "spent_90d",
    "payments_30d",
    "payments_90d",
    "total_discount_30d",
    "mau_mp_1",
    "mau_ml_1",
    "mau_mp_3",
    "mau_ml_3",
    "plata_cuenta_1",
    "plata_cuenta_2",
    "days_since_activity",
]

BINARY_FEATURES = ["has_inversion"]

CATEGORICAL_FEATURES = ["gender", "rango_edad", "tarjetas", "estado"]
