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
        "historical": summarized_stock_data["historical_vol"],
        # "90d": summarized_stock_data["rolling_vol_90d"],
        # "180d": summarized_stock_data["rolling_vol_180d"],
        # "252d": summarized_stock_data["rolling_vol_252d"],
        # "504d": summarized_stock_data["rolling_vol_504d"],
    }
    # spot_price = float(summarized_stock_data["spot_price"])
    strike_price = 500
    barrier = 1.30 * summarized_stock_data["spot_price"]
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
        # plotter = PlotsGenerator(
        #     payload, barrier=barrier, strike=strike_price, label=label
        # )
        # plotter.generate_plots()
        barrier = BarrierOption(
            stock_price=summarized_stock_data["spot_price"],
            risk_free_rate=summarized_stock_data["risk_free_rate"],
            dividend_yield=summarized_stock_data["dividend_yield"],
            volatility=volatility / 100,
            barrier=barrier,
            strike=strike_price,
        )
        print("stock_price:", barrier.stock_price)
        print("risk_free_rate:", barrier.risk_free_rate)
        print("dividend_yield:", barrier.dividend_yield)
        print("volatility:", barrier.volatility)
        print("barrier:", barrier.barrier)
        print("strike:", barrier.strike)
        print("time:", barrier._time)
        print("knocked_in_pct:", payload["knocked_in_pct"])

        monte_carlo_solutions[label] = payload["fair_value"]
        closed_form_solutions[label] = barrier.up_and_in_call()

    # print(f"Monte Carlo Fair Values {monte_carlo_solutions}")
    print(f"Barrier Call Up and In Fair Values {closed_form_solutions}")
    # final_prices = payload["stock_paths"][payload["knocked_in"], -1]
    # payoffs = np.maximum(final_prices - 500, 0)
    # print("mean payoff:", payoffs.mean())
    # print("median payoff:", np.median(payoffs))
    # print("min payoff:", payoffs.min())
    # print("max payoff:", payoffs.max())
    # print("pct with positive payoff:", (payoffs > 0).mean() * 100)


if __name__ == "__main__":
    main()
