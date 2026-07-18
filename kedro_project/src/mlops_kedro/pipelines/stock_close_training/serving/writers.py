from __future__ import annotations

import polars as pl
import psycopg2
from psycopg2.extras import execute_values

from .connections import _timescale_connection_kwargs
from .constants import (
    CLOSE_MODEL_DATASET_COLUMNS,
    CONVENTIONAL_GAP_TRADING_COLUMNS,
    FEAST_OFFLINE_COLUMNS,
    PECNET_PREPROCESSED_COLUMNS,
    TIMESCALE_CLOSE_MODEL_DATASET_TABLE,
    TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE,
    TIMESCALE_PECNET_PREPROCESSED_TABLE,
    TIMESCALE_TABLE,
    TIMESCALE_WRITE_BATCH_SIZE,
)
from .schemas import (
    _create_timescale_close_model_dataset_table,
    _create_timescale_conventional_gap_trading_table,
    _create_timescale_feature_table,
    _create_timescale_pecnet_preprocessed_table,
)
from .transforms import (
    _fill_close_model_dataset_daily_gaps,
    _fill_model_feature_daily_gaps,
    _iter_polars_row_batches,
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

    columns = [
        column
        for column in CONVENTIONAL_GAP_TRADING_COLUMNS
        if column in df.columns
    ]
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

def _write_pecnet_preprocessed_to_timescale(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    columns = PECNET_PREPROCESSED_COLUMNS
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    update_columns = [
        column
        for column in columns
        if column not in {"row_key"}
    ]
    update_assignments = ", ".join(
        f'"{column}" = EXCLUDED."{column}"' for column in update_columns
    )
    insert_sql = f"""
        INSERT INTO {TIMESCALE_PECNET_PREPROCESSED_TABLE} ({quoted_columns})
        VALUES %s
        ON CONFLICT (row_key, event_timestamp)
        DO UPDATE SET {update_assignments};
    """

    with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
        with connection.cursor() as cursor:
            _create_timescale_pecnet_preprocessed_table(cursor)
            tiers = df.get_column("tier").unique().to_list()
            symbols = df.get_column("symbol").unique().to_list()
            cursor.execute(
                f"""
                DELETE FROM {TIMESCALE_PECNET_PREPROCESSED_TABLE}
                WHERE tier = ANY(%s)
                  AND symbol = ANY(%s);
                """,
                (tiers, symbols),
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
