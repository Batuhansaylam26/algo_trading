
  
  create view "dataops_mlops"."main_query"."read_silver_stock_prices__dbt_tmp" as (
    

-- depends_on: 's3://delta-lake-bucket/silver/stock_prices'

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
from delta_scan('s3://delta-lake-bucket/silver/stock_prices')
  );
