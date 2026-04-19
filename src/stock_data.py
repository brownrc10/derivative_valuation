from dataclasses import dataclass, field
from pathlib import Path
import duckdb
import httpx
import pandas as pd
import numpy as np


@dataclass
class StockData:
    """Class that calculates and contains all the necessary information for the Knock-In-Barrier Option and Monte-Carlo Simulation.

    Parameters:
        stock_file_path (str):   Directory that points to the VOO stock data at five min intervals.
        valuation_date (str):    Date for the start of the evaluation period for the option.
        dividend_yield (float):  Dividend yield for VOO pulled off of yahoo_finance.
        risk_free_rate (float):  Risk Free Rate default value provided, but pulled from massive api based on valuation date.

    Methods:
        _calculate_historical_volatiliy: Calulates historical volatility and dividend yield rate from past years data .
        _calculate_rolling_volatility: Calculates rolling average from the valuation date
        _calculate_risk_free_rate: Gets data from Massive API and converts it to a continous rate.
        summarize: Executes all the above tasks and returns a payload with stock information.
    """

    stock_file_path: str
    valuation_date: str
    dividend_yield: float = field(default=0.0109)
    risk_free_rate: float = field(default=0.0)

    def __post_init__(self):
        self.summary: dict = {}

    def __repr__(self):
        return str(self.summary) if self.summary else "StockData not yet summarized."

    def _calculate_historical_volatility(self) -> pd.DataFrame:
        """Method Calculates historical volatility from previous years data.

        Parameters:
            None:

        Returns:
            result (df): Pandas Data frame that contains the annualized volatility and the dividend yield rate.
        Note:
            SQL Flow
            - Read in data and sort.
            - Compute the the log returns via a window function LN(curr_row/prev row)
            - Manage Close Price and Next Day Opening
            - Calculate annualized volatility and retrieve spot price from valuation date
            - Calculate dividend yield rate
        """
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

    def _calculate_rolling_volatility(self, window_days: int) -> pd.DataFrame:
        """Method: Calculates the volatility over a rolling period
        Parameters:
            window_days (int): Number of days to calculate the rolling volatility (ex: Vol last 30 days)
        Returns:
            result (df): Dataframe containing rolling volatility
        Note:
            SQL flow similar to the historical volatility method.
        """
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
        """Method calculates the continous risk free rate after retrieving nominal interst rate.

        Parameters:
            BASE_URL(str): Environment variable containing API URL
            API_KEY(str): Environment variable containing API key
        Returns:
            None:
        Note:
            Gets information for 2-Year Treasury based on the valuation date.
        """
        params = {
            "apiKey": f"{API_KEY}",
            "date": f"{self.valuation_date}",
            "limit": 100,
            "sort": "date.asc",
        }
        with httpx.Client(params=params) as client:
            response = client.get(f"{BASE_URL}")
            data = response.json()
            nominal_rate = data["results"][0]["yield_2_year"] / 100
            self.risk_free_rate = 2 * np.log(1 + nominal_rate / 2)

    def summarize(self, BASE_URL: str, API_KEY: str) -> dict:
        """Method: Executes the internal function to generate StockInformation

        Parameter:
            BASE_URL: URL for API needed to calculate the RFR.
            API_KEY:  KEY for API
        Returns:
            summary (dict): Dictionary containing volatilities values, RFR, Dividend Rate, and Spot Price

        """
        hist = self._calculate_historical_volatility()
        rolling = pd.concat(
            [self._calculate_rolling_volatility(d) for d in [90, 180, 252, 504]]
        )
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
