import os
from pathlib import Path
from dotenv import load_dotenv
from src.stock_data import StockData
from src.barrier_option import BarrierOption
from src.montecarlo import BarrierMonteCarloSimulation

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")


def main():
    stock_data_file_path = Path("parquet_output/ticker_data/five_minute_stock_data")
    stock_data = StockData(
        stock_file_path=stock_data_file_path, valuation_date="2024-04-01"
    )
    summarized_stock_data = stock_data.summarize(BASE_URL=BASE_URL, API_KEY=API_KEY)
    (
        valuation_date,
        spot_price,
        dividend_yield,
        risk_free_rate,
        hist_vol,
        rolling_vol_90,
        rolling_vol_180,
        rolling_vol_252,
        rolling_vol_504,
    ) = summarized_stock_data.values()

    monte_carlo_simulation = BarrierMonteCarloSimulation(
        stock_price=spot_price,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=hist_vol,
        barrier=600,
        strike=500,
        n_simulations=25_000,
    )
    value = monte_carlo_simulation.simulation()
    barrier = BarrierOption(
        stock_price=spot_price,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=hist_vol,
        barrier=400,
        strike=400,
    )
    print(barrier.barrier_call())


if __name__ == "__main__":
    main()
