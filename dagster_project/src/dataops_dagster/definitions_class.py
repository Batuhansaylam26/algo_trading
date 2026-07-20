import os
from pathlib import Path

from dagster import Definitions, in_process_executor
from dagster_dbt import DbtCliResource
from dagster_deltalake import S3Config
from dagster_deltalake.config import ClientConfig
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


class DagsterDefinitionsFactory:
    @staticmethod
    def running_in_container() -> bool:
        return Path("/.dockerenv").exists() or Path("/workspaces").exists()

    @staticmethod
    def local_service_url(value: str | None, *, port: int) -> str:
        if not value or value.lower() == "auto":
            host = (
                "host.docker.internal"
                if DagsterDefinitionsFactory.running_in_container()
                else "127.0.0.1"
            )
            return f"http://{host}:{port}"
        if DagsterDefinitionsFactory.running_in_container():
            return value.replace("127.0.0.1", "host.docker.internal").replace(
                "localhost",
                "host.docker.internal",
            )
        return value.replace("host.docker.internal", "127.0.0.1")

    @staticmethod
    def configure_aws_environment(
        *,
        access_key: str,
        secret_key: str,
        region: str,
    ) -> None:
        os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", access_key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret_key)
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        os.environ.setdefault("AWS_ALLOW_HTTP", "true")
        os.environ.setdefault("AWS_S3_ALLOW_UNSAFE_RENAME", "true")
        os.environ.setdefault("AWS_S3_FORCE_PATH_STYLE", "true")

    @staticmethod
    def build_definitions() -> Definitions:
        bucket_name = os.getenv("DELTA_LAKE_S3_BUCKET", "delta-lake-bucket")
        s3_endpoint = DagsterDefinitionsFactory.local_service_url(
            os.getenv("DELTA_LAKE_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT"),
            port=9000,
        )
        s3_access_key = os.getenv("DELTA_LAKE_S3_ACCESS_KEY", "admin")
        s3_secret_key = os.getenv("DELTA_LAKE_S3_SECRET_KEY", "admin1234")
        s3_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        DagsterDefinitionsFactory.configure_aws_environment(
            access_key=s3_access_key,
            secret_key=s3_secret_key,
            region=s3_region,
        )

        return Definitions(
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
                    root_uri=f"s3://{bucket_name}",
                    storage_options=S3Config(
                        access_key_id=s3_access_key,
                        secret_access_key=s3_secret_key,
                        region=s3_region,
                        bucket=bucket_name,
                        endpoint=s3_endpoint,
                        virtual_hosted_style_request="false",
                        allow_unsafe_rename=True,
                    ),
                    client_options=ClientConfig(allow_http=True),
                ),
            },
            executor=in_process_executor,
        )
