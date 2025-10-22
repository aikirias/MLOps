include .env
export $(shell sed -n 's/=.*//p' .env)

COMPOSE=docker compose --env-file .env -f docker/docker-compose.yml

.PHONY: up down airflow-init seed ps

up:
	$(COMPOSE) up -d

airflow-init:
	$(COMPOSE) run --rm airflow-webserver airflow db init
	$(COMPOSE) run --rm airflow-webserver airflow users create --username admin --password admin --firstname Airflow --lastname Admin --role Admin --email admin@example.com

seed:
	$(COMPOSE) exec -T postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f /docker-entrypoint-initdb.d/02_seed_crypto.sql
	$(COMPOSE) exec -T postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f /docker-entrypoint-initdb.d/03_seed_marketplace.sql

down:
	$(COMPOSE) down

ps:
	$(COMPOSE) ps
