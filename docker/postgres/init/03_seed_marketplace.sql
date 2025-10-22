CREATE TABLE IF NOT EXISTS raw.marketplace_data (
    cus_cust_id_buy BIGINT,
    spent_ml NUMERIC,
    recency_ml DATE,
    frequency_ml INTEGER
);
TRUNCATE TABLE raw.marketplace_data;
COPY raw.marketplace_data (cus_cust_id_buy, spent_ml, recency_ml, frequency_ml)
FROM '/data/MARKETPLACE_DATA.csv'
WITH (FORMAT csv, HEADER true, NULL '', DELIMITER ',');

CREATE TABLE IF NOT EXISTS raw.payments (
    fecha DATE,
    cus_cust_id_sel BIGINT,
    cus_cust_id_buy BIGINT,
    spent NUMERIC,
    tpv_segment_detail TEXT,
    descuento NUMERIC
);
TRUNCATE TABLE raw.payments;
COPY raw.payments (fecha, cus_cust_id_sel, cus_cust_id_buy, spent, tpv_segment_detail, descuento)
FROM '/data/PAYMENTS.csv'
WITH (FORMAT csv, HEADER true, NULL '', DELIMITER ',');

CREATE TABLE IF NOT EXISTS raw.active_user (
    cus_cust_id_buy BIGINT,
    mau_mp_3 NUMERIC,
    mau_ml_3 NUMERIC,
    mau_mp_2 NUMERIC,
    mau_ml_2 NUMERIC,
    mau_mp_1 NUMERIC,
    mau_ml_1 NUMERIC,
    last_login_mp_date_1 DATE,
    last_login_ml_date_1 DATE
);
TRUNCATE TABLE raw.active_user;
COPY raw.active_user (cus_cust_id_buy, mau_mp_3, mau_ml_3, mau_mp_2, mau_ml_2, mau_mp_1, mau_ml_1, last_login_mp_date_1, last_login_ml_date_1)
FROM '/data/ACTIVE_USER.csv'
WITH (FORMAT csv, HEADER true, NULL '', DELIMITER ',');

CREATE TABLE IF NOT EXISTS raw.demograficos (
    city TEXT,
    cus_cust_id_buy BIGINT,
    gender TEXT,
    rango_edad TEXT,
    tarjetas TEXT,
    estado TEXT
);
TRUNCATE TABLE raw.demograficos;
COPY raw.demograficos (city, cus_cust_id_buy, gender, rango_edad, tarjetas, estado)
FROM '/data/DEMOGRAFICOS.csv'
WITH (FORMAT csv, HEADER true, NULL '', DELIMITER ',');

CREATE TABLE IF NOT EXISTS raw.dinero_cuenta (
    cus_cust_id_buy NUMERIC,
    plata_cuenta_1 NUMERIC,
    plata_cuenta_2 NUMERIC,
    inversion TEXT
);
TRUNCATE TABLE raw.dinero_cuenta;
COPY raw.dinero_cuenta (cus_cust_id_buy, plata_cuenta_1, plata_cuenta_2, inversion)
FROM '/data/DINERO_CUENTA.csv'
WITH (FORMAT csv, HEADER true, NULL '', DELIMITER ',');

CREATE TABLE IF NOT EXISTS raw.evaluate (
    cus_cust_id_buy BIGINT,
    churn NUMERIC
);
TRUNCATE TABLE raw.evaluate;
COPY raw.evaluate (cus_cust_id_buy, churn)
FROM '/data/EVALUATE.csv'
WITH (FORMAT csv, HEADER true, NULL '', DELIMITER ',');
