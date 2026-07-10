{{ config(materialized="view", schema="marts") }}

select
    symbol,
    year,
    month,
    day,
    cast(date as date) as trading_date,
    min(low) as low,
    max(high) as high,
    avg(open) as avg_open,
    avg(close) as avg_close,
    sum(volume) as total_volume,
    count(*) as bar_count
from {{ ref("read_silver_stock_prices") }}
group by 1, 2, 3, 4, 5
