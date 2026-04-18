from dataclasses import dataclass, field
from pathlib import Path
import duckdb


@dataclass
class StockData:
    stock_file_path: str
    treasury_file_path: str
    closing_date: str
    dividend_yield: float = field(default=0.0109)

    def calculate_historical_volatility(self):
        with duckdb.connect() as conn:
            sql = f"""
                WITH agg_files AS (
                    SELECT *
                    FROM read_parquet('{self.stock_file_path}/VOO_*_5min.parquet')
                    ORDER BY window_start
                ),
                log_returns AS (
                    SELECT
                        window_start,
                        close,
                        LAG(DATE(window_start)) OVER (ORDER BY window_start) AS prev_date,
                        LN(close / LAG(close) OVER (ORDER BY window_start)) AS log_return
                    FROM agg_files
                ),
                filtered_returns AS (
                    SELECT *
                    FROM log_returns
                    WHERE log_return IS NOT NULL
                    AND DATE(window_start) = prev_date
                ),
                params AS (
                    SELECT
                        STDDEV(log_return) * SQRT(252 * 78) * 100 AS annualized_vol,
                        LAST(close ORDER BY window_start)
                            FILTER (WHERE DATE(window_start) = '{self.closing_date}') AS spot_price
                    FROM filtered_returns
                )
                SELECT
                    annualized_vol,
                    ({self.dividend_yield} * 4 / spot_price) * 100 AS dividend_yield
                FROM params
            """
            result = conn.execute(sql).df()
            return result

    def calculate_rolling_volatility(self, window_days: int):
        with duckdb.connect() as conn:
            sql = f"""
                WITH agg_files AS (
                    SELECT *
                    FROM read_parquet('{self.stock_file_path}/VOO_*_5min.parquet')
                    WHERE DATE(window_start) >= (DATE '{self.closing_date}' - INTERVAL '{window_days} days')
                    AND DATE(window_start) <= DATE '{self.closing_date}'
                    ORDER BY window_start
                ),
                log_returns AS (
                    SELECT
                        window_start,
                        close,
                        LAG(DATE(window_start)) OVER (ORDER BY window_start) AS prev_date,
                        LN(close / LAG(close) OVER (ORDER BY window_start)) AS log_return
                    FROM agg_files
                ),
                filtered_returns AS (
                    SELECT *
                    FROM log_returns
                    WHERE log_return IS NOT NULL
                    AND DATE(window_start) = prev_date
                )
                SELECT
                    {window_days} AS window_days,
                    STDDEV(log_return) * SQRT(252 * 78) * 100 AS annualized_vol
                FROM filtered_returns
            """
            result = conn.execute(sql).df()
            return result

    def calculate_risk_free_rate(self):
        pass
