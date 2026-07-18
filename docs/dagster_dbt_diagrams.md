# Dagster And dbt Diagrams

## Dagster Asset Graph

```mermaid
flowchart LR
    YQ["YahooQuery API"]
    BDaily["bronze/stock_prices\ninterval=1d"]
    BWeekly["bronze/stock_prices_weekly\ninterval=1wk"]
    SDaily["silver/stock_prices\nPandera validated"]
    SWeekly["silver/stock_prices_weekly\nPandera validated"]
    DBT["dbt_models\nDagster dbt asset"]
    DuckDB["DuckDB query + marts"]
    MinIO["MinIO Delta Lake\ns3://delta-lake-bucket"]

    YQ --> BDaily
    YQ --> BWeekly
    BDaily --> SDaily
    BWeekly --> SWeekly
    SDaily --> DBT
    DBT --> DuckDB
    BDaily -. "Delta IO Manager" .-> MinIO
    BWeekly -. "Delta IO Manager" .-> MinIO
    SDaily -. "Delta IO Manager" .-> MinIO
    SWeekly -. "Delta IO Manager" .-> MinIO
```

## Dagster Runtime UML

```mermaid
classDiagram
    direction LR

    class Definitions {
        +assets
        +resources
        +executor
    }

    class BronzeAssets {
        +stock_prices()
        +stock_prices_weekly()
    }

    class SilverAssets {
        +stock_prices()
        +stock_prices_weekly()
        +STOCK_PRICE_SCHEMA
    }

    class DbtAssets {
        +dbt_models()
        +_ensure_manifest_path()
        +_run_dbt_parse()
    }

    class MyDeltaLakeIOManager {
        +type_handlers()
    }

    class DbtCliResource {
        +project_dir
        +profiles_dir
    }

    class DeltaLakePolarsTypeHandler
    class S3Config

    Definitions --> BronzeAssets : registers
    Definitions --> SilverAssets : registers
    Definitions --> DbtAssets : registers
    Definitions --> MyDeltaLakeIOManager : delta_io_manager
    Definitions --> DbtCliResource : dbt resource
    MyDeltaLakeIOManager --> DeltaLakePolarsTypeHandler : handles Polars frames
    MyDeltaLakeIOManager --> S3Config : MinIO/S3 options
    SilverAssets --> BronzeAssets : AssetIn dependencies
    DbtAssets --> DbtCliResource : dbt build
```

## dbt Model Lineage

```mermaid
flowchart LR
    SilverSource["source('silver', 'stock_prices')\ns3://delta-lake-bucket/silver/stock_prices"]
    ReadSilver["read_silver_stock_prices\nschema=query\nview"]
    DailyMart["stock_price_daily\nschema=marts\nview"]
    DuckDBFile["dataops_mlops.duckdb"]
    MinIO["MinIO Delta Lake"]

    MinIO --> SilverSource
    SilverSource -->|"delta_scan()"| ReadSilver
    ReadSilver --> DailyMart
    ReadSilver -. "DuckDB relation" .-> DuckDBFile
    DailyMart -. "DuckDB relation" .-> DuckDBFile
```

## dbt ER

```mermaid
erDiagram
    SILVER_STOCK_PRICES {
        string symbol PK
        datetime date PK
        integer year
        integer month
        integer day
        float open
        float high
        float low
        float close
        float volume
    }

    READ_SILVER_STOCK_PRICES {
        string symbol PK
        datetime date PK
        integer year
        integer month
        integer day
        float open
        float high
        float low
        float close
        float volume
    }

    STOCK_PRICE_DAILY {
        string symbol PK
        date trading_date PK
        integer year
        integer month
        integer day
        float low
        float high
        float avg_open
        float avg_close
        float total_volume
        integer bar_count
    }

    SILVER_STOCK_PRICES ||--|| READ_SILVER_STOCK_PRICES : delta_scan
    READ_SILVER_STOCK_PRICES ||--o{ STOCK_PRICE_DAILY : daily_aggregation
```

## Operational Notes

- Dagster writes daily and weekly bronze/silver Delta tables to MinIO through
  `MyDeltaLakeIOManager`.
- dbt currently reads the daily silver table only:
  `s3://delta-lake-bucket/silver/stock_prices`.
- Weekly silver is consumed by Kedro tier5 feature engineering, not by the
  current dbt marts.
- dbt writes query/mart views into DuckDB using `DBT_DUCKDB_PATH`.
