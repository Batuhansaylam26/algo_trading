from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from math import pi
from pathlib import Path

import pandas as pd
import polars as pl
import psycopg2
from feast import FeatureStore
from psycopg2.extras import execute_values

from ..features.feature_sets import (
    CLOSE_MODEL_DATASET_COLUMNS,
    CLOSE_MODEL_TIME_FEATURE_COLUMNS,
    CONDITION_COLUMNS,
    CONVENTIONAL_GAP_TRADING_COLUMNS,
    FEAST_OFFLINE_COLUMNS,
    FOURIER_TIME_ENCODING_COLUMNS,
    MODEL_TIER_FEATURE_COLUMNS,
    TIER_1_FEATURE_COLUMNS,
    TIER_2_FEATURE_COLUMNS,
)


FEATURE_REPO_DIR = Path(
    os.getenv(
        "FEATURE_REPO_DIR",
        str(Path(__file__).resolve().parents[6] / "feature_repo"),
    )
).resolve()
TIMESCALE_TABLE = "feature_store.stock_model_features"
TIMESCALE_CLOSE_MODEL_DATASET_TABLE = "feature_store.stock_close_model_dataset"
TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE = "feature_store.conventional_gap_trading"
TIMESCALE_WRITE_BATCH_SIZE = int(os.getenv("TIMESCALE_WRITE_BATCH_SIZE", "500"))
TIMESCALE_DAILY_FILL_FREQ = os.getenv("TIMESCALE_DAILY_FILL_FREQ", "B")


def _ensure_feature_repo_on_path() -> None:
    (FEATURE_REPO_DIR / "data").mkdir(parents=True, exist_ok=True)
    feature_repo_parent = str(FEATURE_REPO_DIR.parent)
    if feature_repo_parent not in sys.path:
        sys.path.insert(0, feature_repo_parent)


def _timescale_connection_kwargs() -> dict[str, str | int]:
    return {
        "host": os.getenv("TIMESCALE_HOST", "host.docker.internal"),
        "port": int(os.getenv("TIMESCALE_PORT", "5432")),
        "dbname": os.getenv("TIMESCALE_DB", "dataops"),
        "user": os.getenv("TIMESCALE_USER", "dataops"),
        "password": os.getenv("TIMESCALE_PASSWORD", "dataops"),
    }


def _schema_name(table_name: str) -> str:
    return table_name.split(".", maxsplit=1)[0]


def _create_timescale_feature_table(cursor) -> None:
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {_schema_name(TIMESCALE_TABLE)};")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TIMESCALE_TABLE} (
            symbol TEXT NOT NULL,
            "date" TIMESTAMPTZ NOT NULL,
            created_timestamp TIMESTAMPTZ NOT NULL,
            prev_open DOUBLE PRECISION,
            prev_close DOUBLE PRECISION,
            prev_high DOUBLE PRECISION,
            prev_low DOUBLE PRECISION,
            prev_volume DOUBLE PRECISION,
            calendar_gap_days INTEGER,
            month_sin_1 DOUBLE PRECISION,
            month_cos_1 DOUBLE PRECISION,
            month_sin_2 DOUBLE PRECISION,
            month_cos_2 DOUBLE PRECISION,
            day_sin_1 DOUBLE PRECISION,
            day_cos_1 DOUBLE PRECISION,
            day_sin_2 DOUBLE PRECISION,
            day_cos_2 DOUBLE PRECISION,
            day_of_year_sin_1 DOUBLE PRECISION,
            day_of_year_cos_1 DOUBLE PRECISION,
            day_of_year_sin_2 DOUBLE PRECISION,
            day_of_year_cos_2 DOUBLE PRECISION,
            PRIMARY KEY (symbol, "date")
        );
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE {TIMESCALE_TABLE}
            ADD COLUMN IF NOT EXISTS prev_open DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS prev_close DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS prev_high DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS prev_low DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS prev_volume DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS calendar_gap_days INTEGER,
            ADD COLUMN IF NOT EXISTS month_sin_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS month_cos_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS month_sin_2 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS month_cos_2 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_sin_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_cos_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_sin_2 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_cos_2 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_of_year_sin_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_of_year_cos_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_of_year_sin_2 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_of_year_cos_2 DOUBLE PRECISION;
        """
    )
    cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    cursor.execute(
        """
        SELECT create_hypertable(
            %s,
            'date',
            if_not_exists => TRUE
        );
        """,
        (TIMESCALE_TABLE,),
    )


def _create_timescale_close_model_dataset_table(cursor) -> None:
    cursor.execute(
        f"CREATE SCHEMA IF NOT EXISTS {_schema_name(TIMESCALE_CLOSE_MODEL_DATASET_TABLE)};"
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TIMESCALE_CLOSE_MODEL_DATASET_TABLE} (
            unique_id TEXT NOT NULL,
            ds TIMESTAMPTZ NOT NULL,
            y DOUBLE PRECISION NOT NULL,
            month_sin_1 DOUBLE PRECISION,
            month_cos_1 DOUBLE PRECISION,
            day_sin_1 DOUBLE PRECISION,
            day_cos_1 DOUBLE PRECISION,
            day_of_year_sin_1 DOUBLE PRECISION,
            day_of_year_cos_1 DOUBLE PRECISION,
            created_timestamp TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (unique_id, ds)
        );
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE {TIMESCALE_CLOSE_MODEL_DATASET_TABLE}
            ADD COLUMN IF NOT EXISTS month_sin_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS month_cos_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_sin_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_cos_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_of_year_sin_1 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS day_of_year_cos_1 DOUBLE PRECISION;
        """
    )
    cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    cursor.execute(
        """
        SELECT create_hypertable(
            %s,
            'ds',
            if_not_exists => TRUE
        );
        """,
        (TIMESCALE_CLOSE_MODEL_DATASET_TABLE,),
    )


def _postgres_type_for_conventional_gap_column(column: str) -> str:
    if column in {"symbol", "Gap_Type"}:
        return "TEXT"
    if column in {"date", "created_timestamp"}:
        return "TIMESTAMPTZ"
    if column in {"month", "day", "day_of_year", "calendar_gap_days"}:
        return "INTEGER"
    if column in CONDITION_COLUMNS:
        return "BOOLEAN"
    return "DOUBLE PRECISION"


def _create_timescale_conventional_gap_trading_table(cursor) -> None:
    cursor.execute(
        f"CREATE SCHEMA IF NOT EXISTS {_schema_name(TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE)};"
    )
    column_definitions = ",\n            ".join(
        f'"{column}" {_postgres_type_for_conventional_gap_column(column)}'
        for column in CONVENTIONAL_GAP_TRADING_COLUMNS
    )
    alter_columns = ",\n            ".join(
        f'ADD COLUMN IF NOT EXISTS "{column}" {_postgres_type_for_conventional_gap_column(column)}'
        for column in CONVENTIONAL_GAP_TRADING_COLUMNS
        if column not in {"symbol", "date"}
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE} (
            {column_definitions},
            PRIMARY KEY (symbol, "date")
        );
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE {TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE}
            {alter_columns};
        """
    )
    cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    cursor.execute(
        """
        SELECT create_hypertable(
            %s,
            'date',
            if_not_exists => TRUE
        );
        """,
        (TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE,),
    )


def _to_pandas_for_feature_store(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.select(FEAST_OFFLINE_COLUMNS).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"], utc=True)
    pdf["created_timestamp"] = pd.to_datetime(pdf["created_timestamp"], utc=True)
    return pdf.where(pd.notnull(pdf), None)


def _model_feature_date_key(date: pd.Series) -> pd.Series:
    return pd.to_datetime(date, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _model_feature_key(symbol: pd.Series, date_key: pd.Series) -> pd.Series:
    return symbol.astype(str) + "|" + date_key.astype(str)


def _to_pandas_for_tier2_feature_dataset(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.select(FEAST_OFFLINE_COLUMNS).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"], utc=True)
    pdf["date_key"] = _model_feature_date_key(pdf["date"])
    pdf["feature_key"] = _model_feature_key(pdf["symbol"], pdf["date_key"])
    pdf["created_timestamp"] = pd.to_datetime(pdf["created_timestamp"], utc=True)
    return pdf.where(pd.notnull(pdf), None)


def _iter_polars_row_batches(
    df: pl.DataFrame,
    columns: list[str],
    batch_size: int = TIMESCALE_WRITE_BATCH_SIZE,
):
    selected = df.select(columns)
    for batch in selected.iter_slices(n_rows=batch_size):
        rows = list(batch.iter_rows(named=False))
        if rows:
            yield rows


def _fill_interval() -> str:
    if TIMESCALE_DAILY_FILL_FREQ.upper() == "B":
        return "1d"
    return TIMESCALE_DAILY_FILL_FREQ


def _time_encoding_expressions(
    time_column: str,
    output_columns: list[str],
) -> list[pl.Expr]:
    expressions = []
    time_parts = {
        "month": (pl.col(time_column).dt.month(), 12.0),
        "day": (pl.col(time_column).dt.day(), 31.0),
        "day_of_year": (pl.col(time_column).dt.ordinal_day(), 366.0),
    }

    for column_name, (time_expr, period) in time_parts.items():
        for harmonic in (1, 2):
            angle = 2.0 * pi * harmonic * time_expr.cast(pl.Float64) / period
            sin_column = f"{column_name}_sin_{harmonic}"
            cos_column = f"{column_name}_cos_{harmonic}"
            if sin_column in output_columns:
                expressions.append(angle.sin().alias(sin_column))
            if cos_column in output_columns:
                expressions.append(angle.cos().alias(cos_column))

    return expressions


def _fill_daily_gaps(
    df: pl.DataFrame,
    *,
    id_column: str,
    time_column: str,
    output_columns: list[str],
    preserve_calendar_gap_days: bool = False,
) -> pl.DataFrame:
    if df.is_empty():
        return df.select(output_columns)

    source = (
        df.select(output_columns)
        .with_columns(
            pl.col(time_column)
            .cast(pl.Datetime("us"), strict=False)
            .dt.truncate("1d")
            .alias(time_column)
        )
        .with_row_index("_source_order")
        .sort([id_column, time_column, "_source_order"])
        .unique(subset=[id_column, time_column], keep="last", maintain_order=True)
        .drop("_source_order")
        .with_columns(pl.lit(True).alias("_source_row"))
    )

    date_grid = (
        source.group_by(id_column)
        .agg(
            pl.datetime_ranges(
                pl.col(time_column).min(),
                pl.col(time_column).max(),
                interval=_fill_interval(),
            ).alias(time_column)
        )
        .explode(time_column)
    )

    if TIMESCALE_DAILY_FILL_FREQ.upper() == "B":
        date_grid = date_grid.filter(pl.col(time_column).dt.weekday() <= 5)

    filled = (
        date_grid.join(source, on=[id_column, time_column], how="left")
        .with_columns(pl.col("_source_row").is_null().alias("_synthetic_row"))
        .sort([id_column, time_column])
    )
    if preserve_calendar_gap_days and "calendar_gap_days" in output_columns:
        filled = (
            filled.with_columns(
                (~pl.col("_synthetic_row")).alias("_actual_row")
            )
            .with_columns(
                pl.col("_actual_row")
                .cast(pl.Int64)
                .cum_sum()
                .over(id_column)
                .alias("_actual_segment")
            )
            .with_columns(
                (
                    pl.col(time_column)
                    .cum_count()
                    .over([id_column, "_actual_segment"])
                    - 1
                )
                .cast(pl.Int64)
                .alias("_business_gap_run")
            )
        )

    fill_columns = [
        column
        for column in output_columns
        if column not in {id_column, time_column, "created_timestamp"}
        and not (
            preserve_calendar_gap_days
            and column == "calendar_gap_days"
        )
    ]
    fill_expressions = [
        pl.col(column)
        .forward_fill()
        .backward_fill()
        .over(id_column)
        .alias(column)
        for column in fill_columns
    ]

    if preserve_calendar_gap_days and "calendar_gap_days" in output_columns:
        fill_expressions.append(
            pl.when(pl.col("_synthetic_row"))
            .then(pl.col("_business_gap_run"))
            .when(pl.col("_business_gap_run").shift(1).over(id_column) > 0)
            .then(pl.col("_business_gap_run").shift(1).over(id_column))
            .otherwise(pl.col("calendar_gap_days"))
            .fill_null(0)
            .cast(pl.Int32)
            .alias("calendar_gap_days")
        )

    if "created_timestamp" in output_columns:
        fill_expressions.append(
            pl.lit(datetime.now(timezone.utc)).alias("created_timestamp")
        )

    filled = filled.with_columns(fill_expressions)
    time_expressions = _time_encoding_expressions(time_column, output_columns)
    if time_expressions:
        filled = filled.with_columns(time_expressions)

    return filled.select(output_columns)


def _fill_model_feature_daily_gaps(df: pl.DataFrame) -> pl.DataFrame:
    return _fill_daily_gaps(
        df,
        id_column="symbol",
        time_column="date",
        output_columns=FEAST_OFFLINE_COLUMNS,
        preserve_calendar_gap_days=True,
    )


def _fill_close_model_dataset_daily_gaps(df: pl.DataFrame) -> pl.DataFrame:
    return _fill_daily_gaps(
        df,
        id_column="unique_id",
        time_column="ds",
        output_columns=CLOSE_MODEL_DATASET_COLUMNS,
    )


def _write_model_features_to_timescale(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    df = _fill_model_feature_daily_gaps(df)
    update_columns = [
        column
        for column in FEAST_OFFLINE_COLUMNS
        if column not in {"symbol", "date"}
    ]
    quoted_columns = ", ".join(f'"{column}"' for column in FEAST_OFFLINE_COLUMNS)
    update_assignments = ", ".join(
        f'"{column}" = EXCLUDED."{column}"' for column in update_columns
    )

    insert_sql = f"""
        INSERT INTO {TIMESCALE_TABLE} ({quoted_columns})
        VALUES %s
        ON CONFLICT (symbol, "date")
        DO UPDATE SET {update_assignments};
    """

    with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
        with connection.cursor() as cursor:
            _create_timescale_feature_table(cursor)
            symbols = df.get_column("symbol").unique().to_list()
            cursor.execute(
                f"""
                DELETE FROM {TIMESCALE_TABLE}
                WHERE symbol = ANY(%s);
                """,
                (symbols,),
            )
            total_rows = 0
            for rows in _iter_polars_row_batches(df, FEAST_OFFLINE_COLUMNS):
                execute_values(
                    cursor,
                    insert_sql,
                    rows,
                    page_size=TIMESCALE_WRITE_BATCH_SIZE,
                )
                total_rows += len(rows)

    return total_rows


def _to_pandas_for_close_model_dataset(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.select(CLOSE_MODEL_DATASET_COLUMNS).to_pandas()
    pdf["ds"] = pd.to_datetime(pdf["ds"], utc=True)
    pdf["ds_key"] = _close_model_ds_key(pdf["ds"])
    pdf["series_key"] = _close_model_series_key(pdf["unique_id"], pdf["ds_key"])
    pdf["created_timestamp"] = pd.to_datetime(pdf["created_timestamp"], utc=True)
    return pdf.where(pd.notnull(pdf), None)


def _close_model_ds_key(ds: pd.Series) -> pd.Series:
    return pd.to_datetime(ds, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _close_model_series_key(unique_id: pd.Series, ds_key: pd.Series) -> pd.Series:
    return unique_id.astype(str) + "|" + ds_key.astype(str)


def _write_close_model_dataset_to_timescale(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    df = _fill_close_model_dataset_daily_gaps(df)
    columns = CLOSE_MODEL_DATASET_COLUMNS
    quoted_columns = ", ".join(f'"{column}"' for column in columns)

    insert_sql = f"""
        INSERT INTO {TIMESCALE_CLOSE_MODEL_DATASET_TABLE} ({quoted_columns})
        VALUES %s
        ON CONFLICT (unique_id, ds)
        DO UPDATE SET
            y = EXCLUDED.y,
            month_sin_1 = EXCLUDED.month_sin_1,
            month_cos_1 = EXCLUDED.month_cos_1,
            day_sin_1 = EXCLUDED.day_sin_1,
            day_cos_1 = EXCLUDED.day_cos_1,
            day_of_year_sin_1 = EXCLUDED.day_of_year_sin_1,
            day_of_year_cos_1 = EXCLUDED.day_of_year_cos_1,
            created_timestamp = EXCLUDED.created_timestamp;
    """

    with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
        with connection.cursor() as cursor:
            _create_timescale_close_model_dataset_table(cursor)
            unique_ids = df.get_column("unique_id").unique().to_list()
            cursor.execute(
                f"""
                DELETE FROM {TIMESCALE_CLOSE_MODEL_DATASET_TABLE}
                WHERE unique_id = ANY(%s);
                """,
                (unique_ids,),
            )
            total_rows = 0
            for rows in _iter_polars_row_batches(df, columns):
                execute_values(
                    cursor,
                    insert_sql,
                    rows,
                    page_size=TIMESCALE_WRITE_BATCH_SIZE,
                )
                total_rows += len(rows)

    return total_rows


def _write_conventional_gap_trading_to_timescale(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    columns = CONVENTIONAL_GAP_TRADING_COLUMNS
    update_columns = [
        column
        for column in columns
        if column not in {"symbol", "date"}
    ]
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    update_assignments = ", ".join(
        f'"{column}" = EXCLUDED."{column}"' for column in update_columns
    )

    insert_sql = f"""
        INSERT INTO {TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE} ({quoted_columns})
        VALUES %s
        ON CONFLICT (symbol, "date")
        DO UPDATE SET {update_assignments};
    """

    with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
        with connection.cursor() as cursor:
            _create_timescale_conventional_gap_trading_table(cursor)
            total_rows = 0
            for rows in _iter_polars_row_batches(df, columns):
                execute_values(
                    cursor,
                    insert_sql,
                    rows,
                    page_size=TIMESCALE_WRITE_BATCH_SIZE,
                )
                total_rows += len(rows)

    return total_rows


def _apply_model_feature_definitions() -> FeatureStore:
    _ensure_feature_repo_on_path()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))

    from feature_repo.stock_features import (  # noqa: PLC0415
        stock_feature_row,
        stock_model_features_view,
        stock_model_tier2_dataset_service,
        stock_model_tier2_dataset_view,
        stock_model_tier_1_feature_service,
        stock_model_tier_2_feature_service,
        ticker,
    )

    store.apply(
        [
            ticker,
            stock_model_features_view,
            stock_model_tier_1_feature_service,
            stock_model_tier_2_feature_service,
            stock_feature_row,
            stock_model_tier2_dataset_view,
            stock_model_tier2_dataset_service,
        ]
    )
    return store


def _apply_feast_definitions_and_push(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    store = _apply_model_feature_definitions()
    store.write_to_online_store(
        "stock_model_features",
        _to_pandas_for_feature_store(df),
    )
    store.write_to_online_store(
        "stock_model_tier2_dataset",
        _to_pandas_for_tier2_feature_dataset(df),
    )
    return len(df)


def _apply_close_model_dataset_definition() -> None:
    _ensure_feature_repo_on_path()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))

    from feature_repo.stock_features import (  # noqa: PLC0415
        stock_close_model_dataset_service,
        stock_close_model_dataset_view,
        stock_series,
    )

    store.apply(
        [
            stock_series,
            stock_close_model_dataset_view,
            stock_close_model_dataset_service,
        ]
    )


def _apply_close_model_dataset_definition_and_push(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    df = _fill_close_model_dataset_daily_gaps(df)
    _ensure_feature_repo_on_path()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))

    from feature_repo.stock_features import (  # noqa: PLC0415
        stock_close_model_dataset_service,
        stock_close_model_dataset_view,
        stock_series,
    )

    store.apply(
        [
            stock_series,
            stock_close_model_dataset_view,
            stock_close_model_dataset_service,
        ]
    )
    store.write_to_online_store(
        "stock_close_model_dataset",
        _to_pandas_for_close_model_dataset(df),
    )
    return len(df)


def publish_close_model_dataset(df: pl.DataFrame) -> dict[str, object]:
    timescale_rows = _write_close_model_dataset_to_timescale(df)
    feast_online_rows = _apply_close_model_dataset_definition_and_push(df)

    return {
        "timescale_table": TIMESCALE_CLOSE_MODEL_DATASET_TABLE,
        "timescale_rows": timescale_rows,
        "feast_online_rows": feast_online_rows,
        "feast_registry_applied": True,
        "feast_feature_view": "stock_close_model_dataset",
        "model_columns": ["unique_id", "ds", "y", *CLOSE_MODEL_TIME_FEATURE_COLUMNS],
    }


def _read_close_model_entity_rows() -> pd.DataFrame:
    query = f"""
        SELECT unique_id, ds
        FROM {TIMESCALE_CLOSE_MODEL_DATASET_TABLE}
        ORDER BY unique_id, ds;
    """

    try:
        with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
            entity_df = pd.read_sql_query(query, connection)
    except psycopg2.Error as error:
        if "does not exist" in str(error):
            return pd.DataFrame()
        raise

    if entity_df.empty:
        return entity_df

    entity_df["ds"] = pd.to_datetime(entity_df["ds"], utc=True)
    entity_df["ds_key"] = _close_model_ds_key(entity_df["ds"])
    entity_df["series_key"] = _close_model_series_key(
        entity_df["unique_id"],
        entity_df["ds_key"],
    )
    entity_df["event_timestamp"] = entity_df["ds"]
    return entity_df[["series_key", "unique_id", "ds", "event_timestamp"]]


def _read_close_model_online_entity_rows() -> pd.DataFrame:
    query = f"""
        SELECT unique_id, ds
        FROM {TIMESCALE_CLOSE_MODEL_DATASET_TABLE}
        ORDER BY unique_id, ds;
    """

    with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
        entity_df = pd.read_sql_query(query, connection)

    if entity_df.empty:
        return entity_df

    entity_df["ds"] = pd.to_datetime(entity_df["ds"], utc=True)
    entity_df["ds_key"] = _close_model_ds_key(entity_df["ds"])
    entity_df["series_key"] = _close_model_series_key(
        entity_df["unique_id"],
        entity_df["ds_key"],
    )
    return entity_df[["series_key", "unique_id", "ds"]]


def load_stock_close_model_dataset_from_feast() -> pl.DataFrame:
    _apply_close_model_dataset_definition()
    entity_df = _read_close_model_entity_rows()

    if entity_df.empty:
        return pl.DataFrame(
            schema={
                "unique_id": pl.Utf8,
                "ds": pl.Datetime("us"),
                "y": pl.Float64,
                "month_sin_1": pl.Float64,
                "month_cos_1": pl.Float64,
                "day_sin_1": pl.Float64,
                "day_cos_1": pl.Float64,
                "day_of_year_sin_1": pl.Float64,
                "day_of_year_cos_1": pl.Float64,
            }
        )

    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    historical_features = store.get_historical_features(
        entity_df=entity_df[["series_key", "event_timestamp"]],
        features=[
            f"stock_close_model_dataset:{feature_name}"
            for feature_name in ["y", *CLOSE_MODEL_TIME_FEATURE_COLUMNS]
        ],
    ).to_df()

    if "event_timestamp" in historical_features:
        historical_features = historical_features.drop(columns=["event_timestamp"])

    historical_features = historical_features.merge(
        entity_df[["series_key", "unique_id", "ds"]],
        on="series_key",
        how="left",
    )

    return (
        pl.from_pandas(historical_features)
        .select(
            pl.col("unique_id").cast(pl.Utf8),
            pl.col("ds").cast(pl.Datetime("us"), strict=False),
            pl.col("y").cast(pl.Float64, strict=False),
            pl.col("month_sin_1").cast(pl.Float64, strict=False),
            pl.col("month_cos_1").cast(pl.Float64, strict=False),
            pl.col("day_sin_1").cast(pl.Float64, strict=False),
            pl.col("day_cos_1").cast(pl.Float64, strict=False),
            pl.col("day_of_year_sin_1").cast(pl.Float64, strict=False),
            pl.col("day_of_year_cos_1").cast(pl.Float64, strict=False),
        )
        .sort(["unique_id", "ds"])
    )


def _empty_close_model_dataset_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "unique_id": pl.Utf8,
            "ds": pl.Datetime("us"),
            "y": pl.Float64,
            "month_sin_1": pl.Float64,
            "month_cos_1": pl.Float64,
            "day_sin_1": pl.Float64,
            "day_cos_1": pl.Float64,
            "day_of_year_sin_1": pl.Float64,
            "day_of_year_cos_1": pl.Float64,
        }
    )


def _online_entity_batches(entity_df: pd.DataFrame):
    for start in range(0, len(entity_df), TIMESCALE_WRITE_BATCH_SIZE):
        batch_df = entity_df.iloc[start : start + TIMESCALE_WRITE_BATCH_SIZE].copy()
        yield batch_df, batch_df[["series_key"]].to_dict("records")


def load_stock_close_model_dataset_from_feast_online() -> pl.DataFrame:
    _apply_close_model_dataset_definition()
    entity_df = _read_close_model_online_entity_rows()
    if entity_df.empty:
        return _empty_close_model_dataset_frame()

    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    feature_refs = [
        f"stock_close_model_dataset:{feature_name}"
        for feature_name in ["y", *CLOSE_MODEL_TIME_FEATURE_COLUMNS]
    ]
    frames = []
    for batch_df, entity_rows in _online_entity_batches(entity_df):
        online_features = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_df()
        if not online_features.empty:
            online_features = online_features.merge(
                batch_df[["series_key", "unique_id", "ds"]],
                on="series_key",
                how="left",
            )
            frames.append(pl.from_pandas(online_features))

    if not frames:
        return _empty_close_model_dataset_frame()

    return (
        pl.concat(frames, how="vertical_relaxed")
        .select(
            pl.col("unique_id").cast(pl.Utf8),
            pl.col("ds").cast(pl.Datetime("us"), strict=False),
            pl.col("y").cast(pl.Float64, strict=False),
            pl.col("month_sin_1").cast(pl.Float64, strict=False),
            pl.col("month_cos_1").cast(pl.Float64, strict=False),
            pl.col("day_sin_1").cast(pl.Float64, strict=False),
            pl.col("day_cos_1").cast(pl.Float64, strict=False),
            pl.col("day_of_year_sin_1").cast(pl.Float64, strict=False),
            pl.col("day_of_year_cos_1").cast(pl.Float64, strict=False),
        )
        .sort(["unique_id", "ds"])
    )


def load_stock_close_model_dataset_from_redis() -> pl.DataFrame:
    return load_stock_close_model_dataset_from_feast_online()


def load_stock_close_model_dataset_from_timescale() -> pl.DataFrame:
    columns = ["unique_id", "ds", "y", *CLOSE_MODEL_TIME_FEATURE_COLUMNS]
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    query = f"""
        SELECT {quoted_columns}
        FROM {TIMESCALE_CLOSE_MODEL_DATASET_TABLE}
        ORDER BY unique_id, ds;
    """

    rows = []
    try:
        with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                while batch := cursor.fetchmany(TIMESCALE_WRITE_BATCH_SIZE):
                    rows.extend(batch)
    except psycopg2.Error as error:
        if "does not exist" in str(error):
            return _empty_close_model_dataset_frame()
        raise

    if not rows:
        return _empty_close_model_dataset_frame()

    return (
        pl.DataFrame(rows, schema=columns, orient="row")
        .select(
            pl.col("unique_id").cast(pl.Utf8),
            pl.col("ds").cast(pl.Datetime("us"), strict=False),
            pl.col("y").cast(pl.Float64, strict=False),
            pl.col("month_sin_1").cast(pl.Float64, strict=False),
            pl.col("month_cos_1").cast(pl.Float64, strict=False),
            pl.col("day_sin_1").cast(pl.Float64, strict=False),
            pl.col("day_cos_1").cast(pl.Float64, strict=False),
            pl.col("day_of_year_sin_1").cast(pl.Float64, strict=False),
            pl.col("day_of_year_cos_1").cast(pl.Float64, strict=False),
        )
        .sort(["unique_id", "ds"])
    )


def _empty_model_training_dataset_frame(
    feature_columns: list[str],
) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "unique_id": pl.Utf8,
            "ds": pl.Datetime("us"),
            "y": pl.Float64,
            **{column: pl.Float64 for column in feature_columns},
        }
    )


def _read_tier2_online_entity_rows() -> pd.DataFrame:
    query = f"""
        SELECT symbol, "date"
        FROM {TIMESCALE_TABLE}
        ORDER BY symbol, "date";
    """

    with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
        entity_df = pd.read_sql_query(query, connection)

    if entity_df.empty:
        return entity_df

    entity_df["date"] = pd.to_datetime(entity_df["date"], utc=True)
    entity_df["date_key"] = _model_feature_date_key(entity_df["date"])
    entity_df["feature_key"] = _model_feature_key(
        entity_df["symbol"],
        entity_df["date_key"],
    )
    return entity_df[["feature_key", "symbol", "date"]]


def _tier2_online_entity_batches(entity_df: pd.DataFrame):
    for start in range(0, len(entity_df), TIMESCALE_WRITE_BATCH_SIZE):
        batch_df = entity_df.iloc[start : start + TIMESCALE_WRITE_BATCH_SIZE].copy()
        yield batch_df, batch_df[["feature_key"]].to_dict("records")


def load_stock_model_training_dataset_from_feast_online(
    feature_columns: list[str],
) -> pl.DataFrame:
    _apply_close_model_dataset_definition()
    _apply_model_feature_definitions()
    close_dataset = load_stock_close_model_dataset_from_feast_online()
    entity_df = _read_tier2_online_entity_rows()

    if entity_df.empty or close_dataset.is_empty():
        return _empty_model_training_dataset_frame(feature_columns)

    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    feature_refs = [
        f"stock_model_tier2_dataset:{feature_name}"
        for feature_name in feature_columns
    ]
    frames = []
    for batch_df, entity_rows in _tier2_online_entity_batches(entity_df):
        online_features = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_df()
        if not online_features.empty:
            online_features = online_features.merge(
                batch_df[["feature_key", "symbol", "date"]],
                on="feature_key",
                how="left",
            )
            frames.append(pl.from_pandas(online_features))

    if not frames:
        return _empty_model_training_dataset_frame(feature_columns)

    model_features = (
        pl.concat(frames, how="vertical_relaxed")
        .select(
            pl.col("symbol").cast(pl.Utf8).alias("unique_id"),
            pl.col("date").cast(pl.Datetime("us"), strict=False).alias("ds"),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in feature_columns
            ],
        )
        .sort(["unique_id", "ds"])
    )

    return (
        close_dataset.join(
            model_features,
            on=["unique_id", "ds"],
            how="inner",
        )
        .select(["unique_id", "ds", "y", *feature_columns])
        .drop_nulls(["unique_id", "ds", "y", *feature_columns])
        .unique(subset=["unique_id", "ds"], keep="last", maintain_order=True)
        .sort(["unique_id", "ds"])
    )


def load_stock_tier1_model_dataset_from_feast_online() -> pl.DataFrame:
    return load_stock_model_training_dataset_from_feast_online(
        MODEL_TIER_FEATURE_COLUMNS["tier1"]
    )


def load_stock_tier2_model_dataset_from_feast_online() -> pl.DataFrame:
    return load_stock_model_training_dataset_from_feast_online(
        MODEL_TIER_FEATURE_COLUMNS["tier2"]
    )


def get_online_model_features(
    symbols: list[str],
    feature_columns: list[str] | None = None,
) -> pl.DataFrame:
    if not symbols:
        return pl.DataFrame()

    _ensure_feature_repo_on_path()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    feature_columns = feature_columns or TIER_2_FEATURE_COLUMNS
    online_features = store.get_online_features(
        features=[
            f"stock_model_features:{feature_name}"
            for feature_name in feature_columns
        ],
        entity_rows=[
            {"symbol": symbol}
            for symbol in sorted(set(symbols))
        ],
    ).to_df()

    return pl.from_pandas(online_features)


def publish_model_features(df: pl.DataFrame) -> dict[str, object]:
    timescale_rows = _write_model_features_to_timescale(df)
    feast_online_rows = _apply_feast_definitions_and_push(df)

    return {
        "timescale_table": TIMESCALE_TABLE,
        "timescale_rows": timescale_rows,
        "feast_online_rows": feast_online_rows,
        "feast_feature_view": "stock_model_features",
        "tier_1_features": TIER_1_FEATURE_COLUMNS,
        "tier_2_time_features": [
            "calendar_gap_days",
            *FOURIER_TIME_ENCODING_COLUMNS,
        ],
    }


def publish_conventional_gap_trading(df: pl.DataFrame) -> dict[str, object]:
    timescale_rows = _write_conventional_gap_trading_to_timescale(df)
    return {
        "timescale_table": TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE,
        "timescale_rows": timescale_rows,
        "columns": CONVENTIONAL_GAP_TRADING_COLUMNS,
    }
