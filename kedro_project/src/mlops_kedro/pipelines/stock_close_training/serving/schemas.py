from __future__ import annotations

from .connections import _schema_name
from .constants import (
    CONDITION_COLUMNS,
    CONVENTIONAL_GAP_TRADING_COLUMNS,
    FEAST_OFFLINE_COLUMNS,
    TIMESCALE_CLOSE_MODEL_DATASET_TABLE,
    TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE,
    TIMESCALE_PECNET_PREPROCESSED_TABLE,
    TIMESCALE_TABLE,
)


def _postgres_type_for_feature_column(column: str) -> str:
    if column == "symbol":
        return "TEXT"
    if column in {"date", "created_timestamp"}:
        return "TIMESTAMPTZ"
    if column == "calendar_gap_days":
        return "INTEGER"
    return "DOUBLE PRECISION"


def _create_timescale_feature_table(cursor) -> None:
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {_schema_name(TIMESCALE_TABLE)};")
    column_definitions = ",\n            ".join(
        f'"{column}" {_postgres_type_for_feature_column(column)}'
        for column in FEAST_OFFLINE_COLUMNS
    )
    alter_columns = ",\n            ".join(
        f'ADD COLUMN IF NOT EXISTS "{column}" {_postgres_type_for_feature_column(column)}'
        for column in FEAST_OFFLINE_COLUMNS
        if column not in {"symbol", "date"}
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TIMESCALE_TABLE} (
            {column_definitions},
            PRIMARY KEY (symbol, "date")
        );
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE {TIMESCALE_TABLE}
            {alter_columns};
        """
    )
    cursor.execute(f"ALTER TABLE {TIMESCALE_TABLE} DROP COLUMN IF EXISTS prev_close;")
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
    cursor.execute(
        f"""
        ALTER TABLE {TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE}
            DROP COLUMN IF EXISTS prev_close;
        """
    )
    obsolete_columns = [
        "month",
        "day",
        "day_of_year",
        "prev_open",
        "prev_high",
        "prev_low",
        "prev_volume",
        "calendar_gap_days",
        "month_sin_1",
        "month_cos_1",
        "month_sin_2",
        "month_cos_2",
        "day_sin_1",
        "day_cos_1",
        "day_sin_2",
        "day_cos_2",
        "day_of_year_sin_1",
        "day_of_year_cos_1",
        "day_of_year_sin_2",
        "day_of_year_cos_2",
    ]
    for column in obsolete_columns:
        if column not in CONVENTIONAL_GAP_TRADING_COLUMNS:
            cursor.execute(
                f"""
                ALTER TABLE {TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE}
                    DROP COLUMN IF EXISTS "{column}";
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

def _create_timescale_pecnet_preprocessed_table(cursor) -> None:
    cursor.execute(
        f"CREATE SCHEMA IF NOT EXISTS {_schema_name(TIMESCALE_PECNET_PREPROCESSED_TABLE)};"
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TIMESCALE_PECNET_PREPROCESSED_TABLE} (
            row_key TEXT NOT NULL,
            tier TEXT NOT NULL,
            symbol TEXT NOT NULL,
            event_timestamp TIMESTAMPTZ NOT NULL,
            split TEXT NOT NULL,
            split_index INTEGER NOT NULL,
            variable_name TEXT NOT NULL,
            variable_index INTEGER NOT NULL,
            sample_index INTEGER NOT NULL,
            step_index INTEGER NOT NULL,
            value DOUBLE PRECISION,
            target_y DOUBLE PRECISION,
            created_timestamp TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (row_key, event_timestamp)
        );
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE {TIMESCALE_PECNET_PREPROCESSED_TABLE}
            ADD COLUMN IF NOT EXISTS tier TEXT,
            ADD COLUMN IF NOT EXISTS symbol TEXT,
            ADD COLUMN IF NOT EXISTS event_timestamp TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS split TEXT,
            ADD COLUMN IF NOT EXISTS split_index INTEGER,
            ADD COLUMN IF NOT EXISTS variable_name TEXT,
            ADD COLUMN IF NOT EXISTS variable_index INTEGER,
            ADD COLUMN IF NOT EXISTS sample_index INTEGER,
            ADD COLUMN IF NOT EXISTS step_index INTEGER,
            ADD COLUMN IF NOT EXISTS value DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS target_y DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS created_timestamp TIMESTAMPTZ;
        """
    )
    cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    cursor.execute(
        """
        SELECT create_hypertable(
            %s,
            'event_timestamp',
            if_not_exists => TRUE
        );
        """,
        (TIMESCALE_PECNET_PREPROCESSED_TABLE,),
    )
