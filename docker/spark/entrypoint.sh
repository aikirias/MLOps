#!/bin/bash
set -e

: "${SPARK_MODE:=master}"
: "${SPARK_MASTER_HOST:=0.0.0.0}"
: "${SPARK_MASTER_PORT:=7077}"
: "${SPARK_MASTER_WEBUI_PORT:=8080}"
: "${SPARK_WORKER_WEBUI_PORT:=8081}"
: "${SPARK_WORKER_CORES:=2}"
: "${SPARK_WORKER_MEMORY:=2G}"

if [ "$SPARK_MODE" = "master" ]; then
  exec "$SPARK_HOME"/bin/spark-class org.apache.spark.deploy.master.Master \
    --host "$SPARK_MASTER_HOST" \
    --port "$SPARK_MASTER_PORT" \
    --webui-port "$SPARK_MASTER_WEBUI_PORT"
else
  : "${SPARK_MASTER_URL:=spark://spark-master:7077}"
  exec "$SPARK_HOME"/bin/spark-class org.apache.spark.deploy.worker.Worker \
    "$SPARK_MASTER_URL" \
    --cores "$SPARK_WORKER_CORES" \
    --memory "$SPARK_WORKER_MEMORY" \
    --webui-port "$SPARK_WORKER_WEBUI_PORT"
fi
