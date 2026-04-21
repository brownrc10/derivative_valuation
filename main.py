import os
from pathlib import Path
from dotenv import load_dotenv
from src.stock_data import StockData
from src.barrier_option import BarrierOption
from src.montecarlo import BarrierMonteCarloSimulation
from src.plots import PlotsGenerator
import numpy as np
from scipy.stats import norm

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")


def main():
    closed_form_solutions = {}
    monte_carlo_solutions = {}

    stock_data_file_path = Path("parquet_output/ticker_data/five_minute_stock_data")

    stock_data = StockData(
        stock_file_path=stock_data_file_path, valuation_date="2024-04-01"
    )
    summarized_stock_data = stock_data.summarize(BASE_URL=BASE_URL, API_KEY=API_KEY)
    vol_estimates = {
        "Historical": summarized_stock_data["historical_vol"],
        "30d": summarized_stock_data["rolling_vol_30d"],
        "180d": summarized_stock_data["rolling_vol_180d"],
    }
    print(summarized_stock_data["spot_price"])
    print(summarized_stock_data["dividend_yield"])
    strike_price = 500
    barrier = 1.25 * summarized_stock_data["spot_price"]
    for label, volatility in vol_estimates.items():
        monte_carlo_simulation = BarrierMonteCarloSimulation(
            stock_price=summarized_stock_data["spot_price"],
            risk_free_rate=summarized_stock_data["risk_free_rate"],
            dividend_yield=summarized_stock_data["dividend_yield"],
            volatility=volatility / 100,
            barrier=barrier,
            strike=strike_price,
            n_simulations=100_000,
        )
        payload = monte_carlo_simulation.simulation()
        plotter = PlotsGenerator(
            payload, barrier=barrier, strike=strike_price, label=label
        )
        plotter.generate_plots()
        barrier_instance = BarrierOption(
            stock_price=summarized_stock_data["spot_price"],
            risk_free_rate=summarized_stock_data["risk_free_rate"],
            dividend_yield=summarized_stock_data["dividend_yield"],
            volatility=volatility / 100,
            barrier=102,
            strike=strike_price,
        )

        monte_carlo_solutions[label] = payload["fair_value"]
        closed_form_solutions[label] = barrier_instance.up_and_in_call()

    print(f"Monte Carlo Fair Values {monte_carlo_solutions}")
    print(f"Barrier Call Up and In Fair Values {closed_form_solutions}")

    plotter._plot_comparison(closed_form_solutions, monte_carlo_solutions)
    print(vol_estimates)


if __name__ == "__main__":
    main()
