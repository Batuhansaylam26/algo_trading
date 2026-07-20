

-- depends_on: 's3://delta-lake-bucket/silver/stock_prices_weekly'

select
    symbol,
    date,
    extract(year from date)::integer as year,
    extract(month from date)::integer as month,
    extract(day from date)::integer as day,
    extract(week from date)::integer as week,
    open,
    high,
    low,
    close,
    volume
from delta_scan('s3://delta-lake-bucket/silver/stock_prices_weekly')