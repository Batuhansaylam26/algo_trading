import os

from dagster import Definitions, in_process_executor
from dagster_dbt import DbtCliResource
from dagster_deltalake import S3Config
from dotenv import load_dotenv

from .defs.bronze import (
    stock_prices as bronze_stock_prices,
    stock_prices_weekly as bronze_stock_prices_weekly,
)
from .defs.dbt_assets import DBT_PROFILES_DIR, DBT_PROJECT_DIR, dbt_models
from .defs.silver import (
    stock_prices as silver_stock_prices,
    stock_prices_weekly as silver_stock_prices_weekly,
)
from .utils import MyDeltaLakeIOManager


load_dotenv()

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://host.docker.internal:9000")
BUCKET_NAME = os.getenv("DELTA_LAKE_S3_BUCKET", "delta-lake-bucket")


defs = Definitions(
    assets=[
        bronze_stock_prices,
        bronze_stock_prices_weekly,
        silver_stock_prices,
        silver_stock_prices_weekly,
        dbt_models,
    ],
    resources={
        "dbt": DbtCliResource(
            project_dir=DBT_PROJECT_DIR,
            profiles_dir=DBT_PROFILES_DIR,
        ),
        "delta_io_manager": MyDeltaLakeIOManager(
            root_uri=f"s3://{BUCKET_NAME}",
            storage_options=S3Config(
                endpoint=S3_ENDPOINT,
                allow_unsafe_rename=True,
            ),
        ),
    },
    executor=in_process_executor,
)
