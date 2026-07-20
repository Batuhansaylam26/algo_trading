
  
  create view "dataops_mlops"."main_marts"."stock_price_weekly__dbt_tmp" as (
    

select
    symbol,
    year,
    month,
    week,
    cast(date as date) as week_start_date,
    open,
    high,
    low,
    close,
    volume
from "dataops_mlops"."main_query"."read_silver_stock_prices_weekly"
  );
