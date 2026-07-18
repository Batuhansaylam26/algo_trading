{{ config(materialized="view", schema="marts") }}

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
from {{ ref("read_silver_stock_prices_weekly") }}
