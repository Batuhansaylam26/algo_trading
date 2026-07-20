from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from deltalake import DeltaTable, write_deltalake

from .features.model_dataset import CloseModelDatasetBuilder
from .features.conventional_gap_pipeline import ConventionalGapTradingFeatureBuilder
from .features.indicator_pipeline import StockPriceIndicatorFeatureBuilder
from .features.lookback import LookbackFeatureBuilder
from .features.source import StockPriceFeatureSourceBuilder
from .features.time_encoding import FourierTimeEncoder


@dataclass(slots=True)
class StockCloseFeatureEngineering:
    columns_config: dict[str, list[str]]
    time_encoding_config: dict[str, Any]
    bucket: str | None = None
    time_encoder: FourierTimeEncoder = field(init=False)
    source_builder: StockPriceFeatureSourceBuilder = field(init=False)
    lookback_builder: LookbackFeatureBuilder = field(init=False)
    close_model_builder: CloseModelDatasetBuilder = field(init=False)
    indicator_builder: StockPriceIndicatorFeatureBuilder = field(init=False)
    conventional_builder: ConventionalGapTradingFeatureBuilder = field(init=False)

    def __post_init__(self) -> None:
        self.time_encoder = FourierTimeEncoder(self.time_encoding_config)
        self.source_builder = StockPriceFeatureSourceBuilder(self.time_encoder)
        self.lookback_builder = LookbackFeatureBuilder(self.columns_config)
        self.close_model_builder = CloseModelDatasetBuilder(
            columns_config=self.columns_config,
            time_encoding_config=self.time_encoding_config,
            time_encoder=self.time_encoder,
        )
        self.indicator_builder = StockPriceIndicatorFeatureBuilder(
            columns_config=self.columns_config,
            source_builder=self.source_builder,
            lookback_builder=self.lookback_builder,
        )
        self.conventional_builder = ConventionalGapTradingFeatureBuilder(
            columns_config=self.columns_config,
        )

    @staticmethod
    def delta_storage_options() -> dict[str, str]:
        endpoint = os.getenv(
            "DELTA_LAKE_S3_ENDPOINT_URL",
            "http://127.0.0.1:9000",
        )
        return {
            "AWS_ACCESS_KEY_ID": os.getenv("DELTA_LAKE_S3_ACCESS_KEY", "admin"),
            "AWS_SECRET_ACCESS_KEY": os.getenv(
                "DELTA_LAKE_S3_SECRET_KEY",
                "admin1234",
            ),
            "AWS_ENDPOINT_URL": endpoint,
            "AWS_ALLOW_HTTP": os.getenv("AWS_ALLOW_HTTP", "true"),
            "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
            "AWS_S3_FORCE_PATH_STYLE": "true",
        }

    @staticmethod
    def silver_stock_prices_path(bucket: str) -> str:
        return f"s3://{bucket}/silver/stock_prices"

    @staticmethod
    def silver_stock_prices_weekly_path(bucket: str) -> str:
        return f"s3://{bucket}/silver/stock_prices_weekly"

    @staticmethod
    def stock_price_indicator_features_path(bucket: str) -> str:
        return f"s3://{bucket}/feature_engineering/stock_price_indicators"

    @classmethod
    def read_delta_table(cls, path: str) -> pl.DataFrame:
        table = DeltaTable(path, storage_options=cls.delta_storage_options())
        return pl.from_arrow(table.to_pyarrow_table())

    @classmethod
    def write_delta_table(
        cls,
        path: str,
        df: pl.DataFrame,
        mode: str = "overwrite",
    ) -> None:
        schema_mode = "overwrite" if mode == "overwrite" else None
        write_deltalake(
            path,
            df.to_arrow(),
            mode=mode,
            schema_mode=schema_mode,
            storage_options=cls.delta_storage_options(),
        )

    def read_silver_stock_prices(self, bucket: str | None = None) -> pl.DataFrame:
        return self.read_delta_table(
            self.silver_stock_prices_path(self._bucket_or_default(bucket))
        )

    def read_silver_stock_prices_weekly(
        self,
        bucket: str | None = None,
    ) -> pl.DataFrame:
        return self.read_delta_table(
            self.silver_stock_prices_weekly_path(self._bucket_or_default(bucket))
        )

    def _bucket_or_default(self, bucket: str | None = None) -> str:
        resolved_bucket = bucket or self.bucket
        if not resolved_bucket:
            raise ValueError("bucket must be provided either in constructor or method.")
        return resolved_bucket

    def _harmonics(self, key: str) -> tuple[int, ...]:
        return self.time_encoder.harmonics(key)

    def _period(self, column: str) -> float:
        return self.time_encoder.period(column)

    def fourier_terms(
        self,
        column: str,
        period: float,
        harmonics: tuple[int, ...],
    ):
        return self.time_encoder.terms(column, period, harmonics)

    def add_fourier_time_encoding(
        self,
        df: pl.DataFrame,
        *,
        date_column: str,
        harmonics_key: str,
        drop_date_parts: bool = False,
    ) -> pl.DataFrame:
        return self.time_encoder.add(
            df,
            date_column=date_column,
            harmonics_key=harmonics_key,
            drop_date_parts=drop_date_parts,
        )

    def prepare_feature_source(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.source_builder.prepare(df)

    @staticmethod
    def map_by_symbol(df: pl.DataFrame, function) -> pl.DataFrame:
        return StockPriceFeatureSourceBuilder.map_by_symbol(df, function)

    @staticmethod
    def add_calendar_gap_days(df_symbol: pl.DataFrame) -> pl.DataFrame:
        return StockPriceFeatureSourceBuilder.add_calendar_gap_days(df_symbol)

    def add_time_encoding_for_symbol(self, df_symbol: pl.DataFrame) -> pl.DataFrame:
        return self.source_builder.add_time_encoding_for_symbol(df_symbol)

    def add_model_training_tier_columns_for_symbol(
        self,
        df_symbol: pl.DataFrame,
    ) -> pl.DataFrame:
        return self.source_builder.add_model_training_tier_columns_for_symbol(
            df_symbol,
        )

    @staticmethod
    def with_created_timestamp(df: pl.DataFrame) -> pl.DataFrame:
        return StockPriceFeatureSourceBuilder.with_created_timestamp(df)

    @staticmethod
    def drop_rows_with_missing_model_features(
        df: pl.DataFrame,
        model_ready_columns: list[str],
    ) -> pl.DataFrame:
        return StockPriceFeatureSourceBuilder.drop_rows_with_missing_model_features(
            df,
            model_ready_columns,
        )

    def _add_close_model_time_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.add_fourier_time_encoding(
            df,
            date_column="ds",
            harmonics_key="close_model_harmonics",
            drop_date_parts=True,
        )

    @staticmethod
    def _fill_close_model_business_day_gaps(
        df: pl.DataFrame,
        close_model_freq: str,
    ) -> pl.DataFrame:
        return CloseModelDatasetBuilder.fill_business_day_gaps(df, close_model_freq)

    def to_stock_close_model_dataset(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.close_model_builder.build(df)

    def build_stock_close_model_dataset(
        self,
        silver_stock_prices: pl.DataFrame,
    ) -> pl.DataFrame:
        return self.to_stock_close_model_dataset(silver_stock_prices)

    def build_stock_price_indicator_features(
        self,
        silver_stock_prices: pl.DataFrame,
        silver_stock_prices_weekly: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        return self.indicator_builder.build(
            silver_stock_prices,
            silver_stock_prices_weekly,
        )

    def build_stock_model_features(
        self,
        silver_stock_prices: pl.DataFrame,
        silver_stock_prices_weekly: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        return self.indicator_builder.build_model_features(
            silver_stock_prices,
            silver_stock_prices_weekly,
        )

    def build_stock_feature_sets(
        self,
        silver_stock_prices: pl.DataFrame,
        silver_stock_prices_weekly: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        return self.indicator_builder.build_feature_sets(
            silver_stock_prices,
            silver_stock_prices_weekly,
        )

    def build_conventional_gap_trading_features(
        self,
        stock_price_indicator_features: pl.DataFrame,
    ) -> pl.DataFrame:
        return self.conventional_builder.build(stock_price_indicator_features)
