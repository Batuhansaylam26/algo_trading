# YahooQuery Local Lakehouse Revamp


```text
YahooQuery -> Bronze Delta on MinIO
           -> Silver Delta on MinIO with Polars + Pandera validation and calendar enrichment
           -> dbt + DuckDB query layer
           -> Kedro feature-engineering and model-training pipeline with TimescaleDB + Feast + MLflow

Default services:
Redis, TimescaleDB, MinIO, MLflow, and MLflow Postgres.

Optional ML services:
Kedro-Viz runs from the devcontainer on port 4141.
W&B local runs with the `ml-extra` profile.
```

Project layout:

- `dagster_project/src/dataops_dagster`: Dagster assets, Delta IO manager,
  and dbt orchestration.
- `dbt_project`: DuckDB query-layer and marts.
- `feature_repo`: Feast definitions.
- `kedro_project/src/mlops_kedro`: standalone Kedro project package for
  feature engineering, Feast publishing, TimescaleDB writes, and MLForecast
  training.

The dbt YAML tests from the older pipeline are moved into the silver asset:

- `unique_combination_of_columns(symbol, date)` becomes a Pandera dataframe check.
- `high >= low` becomes a Pandera dataframe check.
- `not_null` checks become non-null Pandera columns.
- positive price checks become Pandera column checks.
- duplicate cleanup happens before validation in the silver asset.

Run order:

```bash
docker compose up -d
cp .env.example .env
source .venv/bin/activate
export PYTHONPATH=/workspaces/yahooquery_lakehouse_revamp/dagster_project/src:/workspaces/yahooquery_lakehouse_revamp/kedro_project/src:/opt/dataops_app/src:/opt/kedro_project/src
python scripts/bootstrap_runtime.py
python -m dagster dev -m dataops_dagster.definitions -h 0.0.0.0 -p 3000
```

In Dagster, materialize `bronze/stock_prices` and `silver/stock_prices` first.
Run the Kedro feature-engineering and model-training pipeline after the silver
Delta table exists:

```bash
cd /opt/kedro_project
kedro run --pipeline stock_close_training
```

Kedro defaults live in:

```text
kedro_project/conf/base/parameters.yml
```

The default model-training configuration runs Optuna with 10 trials and verbose
terminal logging enabled.

Then materialize `dbt_models`, or run dbt directly:

```bash
dbt build --profiles-dir /opt/dbt_project --project-dir /opt/dbt_project
```

Kedro-Viz shows the pipeline graph and datasets:

```bash
cd /opt/kedro_project
kedro viz --host 0.0.0.0 --port 4141
```

Kedro-Viz:

```text
http://localhost:4141
```

Materialize the Dagster assets in this order:

```text
bronze/stock_prices -> silver/stock_prices
silver/stock_prices -> Kedro feature_engineering -> feature_engineering/stock_price_indicators
feature_engineering/stock_price_indicators -> Kedro conventional_gap_trading -> TimescaleDB feature_store.conventional_gap_trading
TimescaleDB/Feast model datasets -> Kedro machine_learning
silver/stock_prices -> dbt query models
```

Silver adds calendar columns extracted from the `date` timestamp:

```text
year, month, day
```

Feature engineering runs in Kedro, outside dbt and Dagster:

```text
silver/stock_prices
  -> feature_engineering/stock_price_indicators
```

Kedro exposes three focused pipelines:

- `feature_engineering`: builds `stock_close_model_dataset`, builds
  `feature_engineering/stock_price_indicators`, writes tier 1/tier 2 model
  features to TimescaleDB, and pushes online model features to Redis through
  Feast.
- `conventional_gap_trading`: reads `feature_engineering/stock_price_indicators`,
  calculates condition flags and `Gap_Type`, and upserts the result to
  TimescaleDB table `feature_store.conventional_gap_trading`.
- `machine_learning`: loads the close-model dataset from TimescaleDB/Feast,
  performs train/test split, trains MLForecast models, and logs to MLflow.

Run them separately:

```bash
cd /opt/kedro_project
kedro run --pipeline feature_engineering
kedro run --pipeline conventional_gap_trading
kedro run --pipeline machine_learning
```

Or run the full chain:

```bash
kedro run --pipeline stock_close_training
```

The conventional gap-trading strategy output is not written to DuckDB/dbt.
It is written to TimescaleDB instead:

- TimescaleDB table: `feature_store.conventional_gap_trading`
- Produced by Kedro pipeline: `conventional_gap_trading`

The feature engineering layer also adds time features:

- Fourier/cyclical encodings for `month`, `day`, and `day_of_year`
- `calendar_gap_days`, the number of missing calendar days between two observed
  trading dates for the same ticker

Model-training tiers are prepared in the feature-engineering Delta output:

- Target column: `target_close`
- Tier 1 features: previous bar OHLCV
  `prev_open`, `prev_close`, `prev_high`, `prev_low`, `prev_volume`
- Tier 2 features: Tier 1 plus Fourier time encodings and `calendar_gap_days`

`target_close` stays in the feature-engineering outputs for analytical/model
dataset work. It is not written to the Feast backend.
Feast exposes two feature services:

- `stock_model_tier_1_features_v1`
- `stock_model_tier_2_features_v1`

Feast uses TimescaleDB as a PostgreSQL offline source and Redis as the online
store. Feast's PostgreSQL offline store is read-oriented, so the Kedro pipeline
upserts feature rows into TimescaleDB before pushing the same rows to Redis via
`FeatureStore.write_to_online_store`.

MinIO console:

```text
http://localhost:9001
admin / admin1234
```

MLflow:

```text
http://localhost:5001
```

W&B Cloud:

```text
https://wandb.ai
```

DuckDB UI:

```text
http://localhost:4213
```

If the `duckdb` CLI is missing inside the devcontainer:

```bash
apt-get update
apt-get install -y curl gzip
curl https://install.duckdb.org | sh
ln -sf /root/.duckdb/cli/latest/duckdb /usr/local/bin/duckdb
```

Start it from the DuckDB CLI:

```bash
duckdb /workspaces/yahooquery_lakehouse_revamp/database/duckdb/dataops_mlops.duckdb -ui
```

Or from SQL:

```sql
CALL start_ui_server();
```

The default DuckDB UI port is `4213`; the devcontainer forwards that port.

Feast is wired with:

- TimescaleDB/PostgreSQL offline store table: `feature_store.stock_model_features`
- Redis online store on `localhost:6379`
- Feature services: `stock_model_tier_1_features_v1` and
  `stock_model_tier_2_features_v1`
