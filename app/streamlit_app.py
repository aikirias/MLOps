from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import mlflow.pyfunc
import pandas as pd
import streamlit as st
from PIL import Image

# Feature configuration aligned with ml/model_config.py
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

CATEGORICAL_FEATURES = {
    "gender": ["male", "female", "unknown"],
    "rango_edad": [
        "01.Menor de 18 años",
        "02.Entre 18 y 25 años",
        "03.Entre 26 y 30 años",
        "04.Entre 31 y 35 años",
        "05.Entre 36 y 40 años",
        "06.Entre 41 y 55 años",
        "07.Entre 56 y 65 años",
        "08.Mayor a 65 años",
    ],
    "tarjetas": ["Credit Card", "Debit Card", "Account Money", "Other"],
    "estado": ["capital federal", "buenos aires", "cordoba", "santa fe", "otros"],
}

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-logreg")
MODEL_STAGE = os.getenv("MLFLOW_STREAMLIT_MODEL_STAGE", "Production")


@st.cache_resource(show_spinner=False)
def _load_model() -> Any:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.pyfunc.load_model(model_uri)


def _safe_load_image(name: str) -> Image.Image | None:
    file = ASSETS_DIR / name
    if file.exists():
        try:
            return Image.open(file)
        except Exception:
            return None
    return None


def _extract_probability(output: Any) -> float:
    if isinstance(output, pd.DataFrame):
        if "probability" in output.columns:
            value = output.loc[0, "probability"]
            if hasattr(value, "__getitem__"):
                return float(value[1])
            return float(value)
        if "prediction" in output.columns:
            return float(output.loc[0, "prediction"])
    if isinstance(output, pd.Series):
        return float(output.iloc[0])
    if isinstance(output, (list, tuple)):
        return float(output[0])
    return float(output)


def _predict(payload: Dict[str, List[Any]]) -> float:
    model = _load_model()
    df = pd.DataFrame(payload)
    prediction_output = model.predict(df)
    return _extract_probability(prediction_output)


def _layout_header() -> None:
    st.title("Churn Predictor Dashboard")
    hero = _safe_load_image("high_level_placeholder.png")
    if hero is not None:
        st.image(hero, caption="Pipeline Overview", use_column_width=True)
    st.markdown(
        """
        ### Context
        Configure los atributos recientes del cliente y obtenga la probabilidad de
        churn estimada por el modelo registrado en MLflow. El dashboard se conecta
        directamente al model registry sin utilizar `MLFlowDeploymentService`.
        """
    )


def _layout_feature_table() -> None:
    description = pd.DataFrame(
        {
            "Feature": NUMERIC_FEATURES + BINARY_FEATURES + list(CATEGORICAL_FEATURES.keys()),
            "Descripción": [
                "Monto gastado últimos 7 días",
                "Monto gastado últimos 30 días",
                "Monto gastado últimos 90 días",
                "Cantidad de pagos 30 días",
                "Cantidad de pagos 90 días",
                "Total de descuentos 30 días",
                "MAU Mercado Pago (último mes)",
                "MAU Mercado Libre (último mes)",
                "MAU Mercado Pago (últimos 3 meses)",
                "MAU Mercado Libre (últimos 3 meses)",
                "Saldo disponible en cuenta (1)",
                "Saldo disponible en cuenta (2)",
                "Días desde última actividad",
                "Cliente con productos de inversión",
                "Género",
                "Rango de edad declarado",
                "Tipo principal de medio de pago",
                "Provincia o estado declarado",
            ],
        }
    )
    st.markdown("#### Descripción de características")
    st.dataframe(description, hide_index=True, use_container_width=True)


def _input_sidebar() -> Dict[str, List[Any]]:
    st.sidebar.header("Características del cliente")
    values: Dict[str, List[Any]] = {}

    # Numeric inputs con valores por defecto razonables
    default_numeric = {
        "spent_7d": 50.0,
        "spent_30d": 200.0,
        "spent_90d": 600.0,
        "payments_30d": 3.0,
        "payments_90d": 8.0,
        "total_discount_30d": 10.0,
        "mau_mp_1": 2.0,
        "mau_ml_1": 1.0,
        "mau_mp_3": 5.0,
        "mau_ml_3": 3.0,
        "plata_cuenta_1": 1000.0,
        "plata_cuenta_2": 500.0,
        "days_since_activity": 15.0,
    }

    for feature in NUMERIC_FEATURES:
        values[feature] = [
            st.sidebar.number_input(
                feature,
                value=float(default_numeric.get(feature, 0.0)),
            )
        ]

    # Binary feature (checkbox)
    has_inv = st.sidebar.checkbox("Cliente con inversión activa", value=True)
    values["has_inversion"] = [1 if has_inv else 0]

    # Categorical features (selectboxes)
    for feature, options in CATEGORICAL_FEATURES.items():
        values[feature] = [st.sidebar.selectbox(feature, options, index=0)]

    return values


def main() -> None:
    _layout_header()
    _layout_feature_table()
    payload = _input_sidebar()

    if st.button("Predecir probabilidad de churn"):
        try:
            prediction = _predict(payload)
            churn_score = prediction
            st.success(
                f"Probabilidad estimada de churn: {churn_score:.2%}"
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"No fue posible obtener la predicción: {exc}")

    if st.button("Resultados esperados"):
        st.write(
            "El modelo actual proviene del pipeline mensual (`train_and_register`) que "
            "entrena una regresión logística con los últimos datos de features semanal."
        )
        feature_img = _safe_load_image("feature_importance_placeholder.png")
        if feature_img is not None:
            st.image(feature_img, caption="Importancia aproximada de características")
        else:
            st.info(
                "Agregar materiales gráficos en `app/assets/` para enriquecer la visualización."
            )


if __name__ == "__main__":
    main()
