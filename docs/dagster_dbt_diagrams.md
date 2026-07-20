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
    SWeekly --> DBT
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

    class DagsterDefinitionsFactory {
        +running_in_container()
        +local_service_url()
        +configure_aws_environment()
        +build_definitions()
    }

    class BronzeStockPriceAssets {
        +load_yahooquery_history()
        +stock_price_metadata()
        +stock_prices()
        +stock_prices_weekly()
    }

    class SilverStockPriceAssets {
        +column_gt_zero()
        +column_ge_zero()
        +high_gte_low()
        +unique_symbol_date()
        +standardize_stock_prices()
        +deduplicate_stock_prices()
        +add_calendar_columns()
        +stock_prices()
        +stock_prices_weekly()
        +STOCK_PRICE_SCHEMA
    }

    class DagsterDbtAssets {
        +dbt_models()
        +ensure_manifest_path()
        +run_dbt_parse()
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

    DagsterDefinitionsFactory --> BronzeStockPriceAssets : registers asset objects
    DagsterDefinitionsFactory --> SilverStockPriceAssets : registers asset objects
    DagsterDefinitionsFactory --> DagsterDbtAssets : registers dbt asset object
    DagsterDefinitionsFactory --> MyDeltaLakeIOManager : delta_io_manager
    DagsterDefinitionsFactory --> DbtCliResource : dbt resource
    MyDeltaLakeIOManager --> DeltaLakePolarsTypeHandler : handles Polars frames
    MyDeltaLakeIOManager --> S3Config : MinIO/S3 options
    SilverStockPriceAssets --> BronzeStockPriceAssets : AssetIn dependencies
    DagsterDbtAssets --> DbtCliResource : dbt build
```

Dagster still receives top-level asset objects named `stock_prices`,
`stock_prices_weekly`, and `dbt_models` from facade modules, but their compute
functions live as static methods in one-class implementation modules. This keeps
Dagster discovery stable while matching the project OOP convention.

## dbt Model Lineage

```mermaid
flowchart LR
    SilverDailySource["source('silver', 'stock_prices')\ns3://delta-lake-bucket/silver/stock_prices"]
    SilverWeeklySource["source('silver', 'stock_prices_weekly')\ns3://delta-lake-bucket/silver/stock_prices_weekly"]
    ReadSilverDaily["read_silver_stock_prices\nschema=query\nview"]
    ReadSilverWeekly["read_silver_stock_prices_weekly\nschema=query\nview"]
    DailyMart["stock_price_daily\nschema=marts\nview"]
    WeeklyMart["stock_price_weekly\nschema=marts\nview"]
    DuckDBFile["dataops_mlops.duckdb"]
    MinIO["MinIO Delta Lake"]

    MinIO --> SilverDailySource
    MinIO --> SilverWeeklySource
    SilverDailySource -->|"delta_scan()"| ReadSilverDaily
    SilverWeeklySource -->|"delta_scan()"| ReadSilverWeekly
    ReadSilverDaily --> DailyMart
    ReadSilverWeekly --> WeeklyMart
    ReadSilverDaily -. "DuckDB relation" .-> DuckDBFile
    ReadSilverWeekly -. "DuckDB relation" .-> DuckDBFile
    DailyMart -. "DuckDB relation" .-> DuckDBFile
    WeeklyMart -. "DuckDB relation" .-> DuckDBFile
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

    SILVER_STOCK_PRICES_WEEKLY {
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

    READ_SILVER_STOCK_PRICES_WEEKLY {
        string symbol PK
        datetime date PK
        integer year
        integer month
        integer day
        integer week
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

    STOCK_PRICE_WEEKLY {
        string symbol PK
        date week_start_date PK
        integer year
        integer month
        integer week
        float open
        float high
        float low
        float close
        float volume
    }

    SILVER_STOCK_PRICES ||--|| READ_SILVER_STOCK_PRICES : delta_scan
    READ_SILVER_STOCK_PRICES ||--o{ STOCK_PRICE_DAILY : daily_aggregation
    SILVER_STOCK_PRICES_WEEKLY ||--|| READ_SILVER_STOCK_PRICES_WEEKLY : delta_scan
    READ_SILVER_STOCK_PRICES_WEEKLY ||--o{ STOCK_PRICE_WEEKLY : weekly_projection
```

## Runtime Storage Layout

```mermaid
flowchart LR
    Dagster["Dagster assets"]
    dbt["dbt views"]
    Kedro["Kedro ML pipelines"]
    MinIO["MinIO buckets\nDelta Lake + MLflow model artifacts"]
    DuckDB["DuckDB file\ndataops_mlops.duckdb"]
    Timescale["TimescaleDB\nfeature_store + marts"]
    LocalArtifacts["Git-tracked lightweight artifacts\nartifacts/stock_close_training"]

    Dagster -->|"bronze/silver Delta writes"| MinIO
    dbt -->|"query/mart views"| DuckDB
    Kedro -->|"feature serving tables"| Timescale
    Kedro -->|"MLflow model artifacts"| MinIO
    Kedro -->|"params, metric CSVs, compact plots"| LocalArtifacts
```

## Operational Notes

- Dagster writes daily and weekly bronze/silver Delta tables to MinIO through
  `MyDeltaLakeIOManager`.
- dbt reads both daily and weekly silver Delta tables:
  `s3://delta-lake-bucket/silver/stock_prices` and
  `s3://delta-lake-bucket/silver/stock_prices_weekly`.
- Weekly silver is exposed through `read_silver_stock_prices_weekly` and
  `stock_price_weekly`, both materialized as DuckDB views.
- dbt writes query/mart views into DuckDB using `DBT_DUCKDB_PATH`.
- Lightweight Kedro artifacts under `artifacts/stock_close_training` are
  intentionally Git-trackable. Heavy runtime data remains in MinIO, TimescaleDB,
  or DuckDB instead of the project artifact folder.
