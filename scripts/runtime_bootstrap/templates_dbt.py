from __future__ import annotations


DBT_RUNTIME_FILES = {
    "dbt_project.yml": """name: "dbt_project"
version: "1.0.0"
profile: "dataops_mlops"

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

clean-targets:
  - "target"
  - "dbt_packages"

models:
  dbt_project:
    query_layer:
      +materialized: view
      +schema: query
    marts:
      +materialized: view
      +schema: marts
""",
    "profiles.yml": """dataops_mlops:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "{{ env_var('DBT_DUCKDB_PATH', './dataops_mlops.duckdb') }}"
      threads: 4

      extensions:
        - httpfs
        - delta

      secrets:
        - type: s3
          provider: config
          key_id: "{{ env_var('DELTA_LAKE_S3_ACCESS_KEY', 'admin') }}"
          secret: "{{ env_var('DELTA_LAKE_S3_SECRET_KEY', 'admin1234') }}"
          region: us-east-1
          endpoint: host.docker.internal:9000
          use_ssl: false
          url_style: path
          scope: "s3://{{ env_var('DELTA_LAKE_S3_BUCKET', 'delta-lake-bucket') }}"
""",
    "models/query_layer/read_silver_stock_prices.sql": """{{ config(materialized="view", schema="query") }}

-- depends_on: {{ source("silver", "stock_prices") }}

select
    symbol,
    date,
    extract(year from date)::integer as year,
    extract(month from date)::integer as month,
    extract(day from date)::integer as day,
    open,
    high,
    low,
    close,
    volume
from delta_scan('s3://{{ env_var("DELTA_LAKE_S3_BUCKET", "delta-lake-bucket") }}/silver/stock_prices')
""",
    "models/query_layer/schema.yml": """version: 2

models:
  - name: read_silver_stock_prices
    description: "Query-layer DuckDB view over the validated silver Delta table."
    columns:
      - name: symbol
        description: "Ticker symbol."
      - name: date
        description: "Quote timestamp in UTC without timezone."
      - name: year
        description: "Year extracted from the quote timestamp."
      - name: month
        description: "Month extracted from the quote timestamp."
      - name: day
        description: "Day of month extracted from the quote timestamp."
      - name: open
        description: "Open price."
      - name: high
        description: "High price."
      - name: low
        description: "Low price."
      - name: close
        description: "Close price."
      - name: volume
        description: "Traded volume."
""",
    "models/query_layer/sources.yml": """version: 2

sources:
  - name: silver
    description: "Validated and calendar-enriched Delta Lake silver layer written by Polars and Pandera."
    tables:
      - name: stock_prices
        description: "Cleaned, deduplicated, validated, and calendar-enriched intraday stock prices."
        meta:
          external_location: "s3://delta-lake-bucket/silver/stock_prices"
          format: "delta"
          dagster:
            asset_key:
              - silver
              - stock_prices
""",
    "models/marts/stock_price_daily.sql": """{{ config(materialized="view", schema="marts") }}

select
    symbol,
    year,
    month,
    day,
    cast(date as date) as trading_date,
    min(low) as low,
    max(high) as high,
    avg(open) as avg_open,
    avg(close) as avg_close,
    sum(volume) as total_volume,
    count(*) as bar_count
from {{ ref("read_silver_stock_prices") }}
group by 1, 2, 3, 4, 5
""",
    "models/marts/schema.yml": """version: 2

models:
  - name: stock_price_daily
    description: "Daily stock-price mart built from the silver query-layer view."
    columns:
      - name: symbol
        description: "Ticker symbol."
      - name: year
        description: "Trading year."
      - name: month
        description: "Trading month."
      - name: day
        description: "Trading day of month."
      - name: trading_date
        description: "Trading date."
      - name: low
        description: "Daily low."
      - name: high
        description: "Daily high."
      - name: avg_open
        description: "Average intraday open price."
      - name: avg_close
        description: "Average intraday close price."
      - name: total_volume
        description: "Total daily volume."
      - name: bar_count
        description: "Number of intraday bars in the day."
""",
}
