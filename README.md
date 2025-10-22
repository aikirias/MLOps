# Meli ML Flow

Proyecto de referencia que combina ETL de criptoactivos y un pipeline de churn con prácticas básicas de MLOps. Todo el stack se levanta con `docker-compose` y utiliza Postgres como repositorio de datos, Airflow para la orquestación, MLflow para el registro de modelos y MinIO como store de artefactos.

## Stack principal
- **PostgreSQL 15**: staging, feature store y tablas operativas.
- **Apache Airflow 2.8**: scheduler y webserver (LocalExecutor).
- **MLflow 2.10**: tracking server + model registry con artefactos en MinIO.
- **MinIO**: backend S3 compatible (`s3://mlflow-artifacts/`).
- **JupyterLab**: entorno de exploración y notebooks.
- **Streamlit**: UI para explorar el modelo de churn registrado en MLflow.

## Estructura relevante
```
mp-churn/
├─ docker/
│  ├─ docker-compose.yml        # Stack completo
│  ├─ airflow/Dockerfile        # Imagen custom con dependencias
│  ├─ airflow/requirements.txt  # Dependencias Python para Airflow
│  ├─ mlflow/Dockerfile         # Imagen ligera para el server de MLflow
│  └─ postgres/init/*.sql       # Creación de esquemas + seed de datos
├─ airflow/
│  ├─ dags/
│  │  ├─ etl_crypto_events_d.py       # ETL diario de BT_CRYPTO_EVENTS
│  │  ├─ churn_feature_build_w.py     # Construcción semanal de features
│  │  ├─ churn_train_register_m.py    # Training y registro mensual
│  │  └─ churn_batch_score_w.py       # Scoring batch semanal
│  └─ include/
│     ├─ sql/
│     │  ├─ seed_raw.sql              # Re-seed manual de la tabla raw
│     │  ├─ upsert_crypto_events.sql  # UPSERT + activar registros
│     │  └─ features_churn.sql        # SQL parametrizable de features
│     └─ expectations/                # Placeholder para data checks
├─ ml/
│  ├─ train.py                        # Entrenamiento + registro MLflow
│  ├─ infer_batch.py                  # Scoring semanal
│  ├─ utils_io.py                     # Utilidades de IO
│  ├─ model_config.py                 # Definición de columnas del modelo
│  └─ __init__.py
├─ .env
├─ Makefile
└─ README.md
```

Los CSV requeridos (`MARKETPLACE_DATA.csv`, `PAYMENTS.csv`, `ACTIVE_USER.csv`, `DEMOGRAFICOS.csv`, `DINERO_CUENTA.csv`, `EVALUATE.csv`) deben permanecer en el root del proyecto para el seed automático de Postgres.

## Puesta en marcha
1. Ajustar variables si es necesario en `.env` (credenciales, puertos, etc.).
2. Construir y levantar el stack:
   ```bash
   make up
   ```
3. Inicializar la base de datos de Airflow y crear usuario admin:
   ```bash
   make airflow-init
   ```
4. Re-ejecutar el seed de los datos (opcional si los contenedores son nuevos):
   ```bash
   make seed
   ```

Servicios expuestos:
- Airflow Webserver: http://localhost:8080 (usuario: `admin` / password: `admin`).
- MLflow UI: http://localhost:5000 (puerto configurable en `.env`).
- MinIO Console: http://localhost:${MINIO_CONSOLE_PORT:-9001} (user/pass en `.env`).
- JupyterLab: http://localhost:${JUPYTER_PORT:-8888} (token en logs del contenedor).

## Pipelines en Airflow
- **etl_crypto_events_d** (`@daily`):
  - Carga staging desde `raw.bt_crypto_transaction_history` con conversiones de tipos.
  - Controla integridad (nulos, fechas futuras, purchase_value).
  - UPSERT hacia `ops.bt_crypto_events`.
  - Desactiva (`is_active = false`) registros de los últimos 5 días que no aparezcan en staging.
- **churn_feature_build_w** (`@weekly`):
  - Ejecuta `features_churn.sql` para armar el feature store (`features.churn_weekly_features`).
  - Etiqueta churn combinando `raw.evaluate` y una ventana de inactividad ≥ 60 días.
- **churn_train_register_m** (`@monthly`):
  - Usa las últimas 8 semanas de features para entrenar `LogisticRegression`.
  - Registra el modelo en MLflow y promueve la versión a `Production`.
- **churn_batch_score_w** (`@weekly`):
  - Carga la versión `Production` del modelo desde MLflow.
  - Scorea la cohorte semanal y persiste resultados en `ops.churn_scoring`.

### Ventanas históricas utilizadas
Los CSV provistos contienen actividad principalmente durante 2019, por lo que todos los DAGs están parametrizados con `start_date` en 2019. Para reproducir manualmente el pipeline end-to-end se pueden disparar las corridas:

```bash
# Features semanales (ejemplos)
airflow dags test churn_feature_build_w 2019-05-26

# Entrenamiento mensual
airflow dags test churn_train_register_m 2019-05-31

# Scoring semanal
airflow dags test churn_batch_score_w 2019-05-26
```

La iteración diaria de cripto (`etl_crypto_events_d`) también se alinea con estas fechas sin necesitar parámetros adicionales.

### UI de Streamlit (modelo de churn)
La app de Streamlit (http://localhost:8501) consume el modelo `churn-logreg` registrado por el
pipeline mensual (`ml/train.py`). Para asegurarte de que exista una versión en `Production`
podés ejecutar:

```bash
docker compose --env-file .env -f docker/docker-compose.yml exec airflow-webserver \
  airflow dags test churn_train_register_m 2019-05-31
```

Luego reiniciá el servicio de Streamlit si fuese necesario:

```bash
docker compose --env-file .env -f docker/docker-compose.yml up -d streamlit
```

La interfaz permite cargar manualmente los features semanales definidos en `ml/model_config.py`
y obtiene la probabilidad estimada de churn directamente desde MLflow, sin utilizar
`MLFlowDeploymentService`.

## Notebooks y análisis
El contenedor de Jupyter monta el repo completo en `/home/jovyan/work`. Los notebooks disponen de las mismas variables de conexión que Airflow (`POSTGRES_*`, `MLFLOW_*`).

## Consideraciones adicionales
- Los scripts de inicialización de Postgres crean los esquemas `raw`, `staging`, `features`, `ops` y realizan el `COPY FROM` de todos los CSV.
- `make seed` puede ejecutarse en cualquier momento para reinicializar tablas raw/staging sin recrear contenedores.
- El bucket de MinIO (`mlflow-artifacts`) se crea automáticamente mediante el servicio auxiliar `minio-create-bucket`.
- Ajustar el parámetro `lookback_weeks` en `churn_train_register_m.py` según sea necesario.

¡Listo! Con esto deberías poder orquestar el ETL de cripto y el pipeline de churn end-to-end utilizando Airflow y MLflow.
