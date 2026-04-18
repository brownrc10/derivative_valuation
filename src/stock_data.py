from dataclasses import dataclass, field
from pathlib import Path
import duckdb
import httpx
import pandas as pd


@dataclass
class StockData:
    stock_file_path: str
    valuation_date: str
    dividend_yield: float = field(default=0.0109)
    risk_free_rate: float = field(default=0.0)

    def __post_init__(self):
        self.summary: dict = {}

    def __repr__(self):
        return str(self.summary) if self.summary else "StockData not yet summarized."

    def _calculate_historical_volatility(self):
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
                            FILTER (WHERE DATE(window_start) = '{self.valuation_date}') AS spot_price
                    FROM filtered_returns
                )
                SELECT
                    annualized_vol,
                    ({self.dividend_yield} * 4 / spot_price) * 100 AS dividend_yield,
                    spot_price
                FROM params
            """
            result = conn.execute(sql).df()
            return result

    def _calculate_rolling_volatility(self, window_days: int):
        with duckdb.connect() as conn:
            sql = f"""
                WITH agg_files AS (
                    SELECT *
                    FROM read_parquet('{self.stock_file_path}/VOO_*_5min.parquet')
                    WHERE DATE(window_start) >= (DATE '{self.valuation_date}' - INTERVAL '{window_days} days')
                    AND DATE(window_start) <= DATE '{self.valuation_date}'
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
                    STDDEV(log_return) * SQRT(252 * 78) * 100 AS rolling_vol
                FROM filtered_returns
            """
            result = conn.execute(sql).df()
            return result

    def _calculate_risk_free_rate(self, BASE_URL: str, API_KEY: str):
        params = {
            "apiKey": f"{API_KEY}",
            "date": "2024-04-01",
            "limit": 100,
            "sort": "date.asc",
        }
        with httpx.Client(params=params) as client:
            response = client.get(f"{BASE_URL}")
            data = response.json()
            self.risk_free_rate = data["results"][0]["yield_2_year"]
            print(self.risk_free_rate)

    def summarize(self, BASE_URL: str, API_KEY: str) -> dict:
        hist = self._calculate_historical_volatility()
        rolling = pd.concat(
            [self._calculate_rolling_volatility(d) for d in [90, 180, 252, 504]]
        )
        print(BASE_URL)
        self._calculate_risk_free_rate(BASE_URL, API_KEY)

        self.summary = {
            "valuation_date": self.valuation_date,
            "spot_price": hist["spot_price"].iloc[0],
            "dividend_yield": hist["dividend_yield"].iloc[0],
            "risk_free_rate": self.risk_free_rate,
            "historical_vol": hist["annualized_vol"].iloc[0],
            "rolling_vol_90d": rolling.loc[
                rolling["window_days"] == 90, "rolling_vol"
            ].iloc[0],
            "rolling_vol_180d": rolling.loc[
                rolling["window_days"] == 180, "rolling_vol"
            ].iloc[0],
            "rolling_vol_252d": rolling.loc[
                rolling["window_days"] == 252, "rolling_vol"
            ].iloc[0],
            "rolling_vol_504d": rolling.loc[
                rolling["window_days"] == 504, "rolling_vol"
            ].iloc[0],
        }
        return self.summary
