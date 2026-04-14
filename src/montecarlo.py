import numpy as np
from typing import Tuple


class MonteCarloSimulation:
    """A class that preforms the MonteCarlo simulation

    Parameters:
        stock_price (float): The stock price at the opening of the Performance Period
        risk_free_rate (float): Risk Free Rate calculated using the 3 Year Treasury Yield
        dividend_yield (float): Dividend Yield of Hershey Stock
        volatility (float): Volatility Calculation Based of Historical Stock Price Data
        reward_price (float): Price that the stock must be over for 20 consecutive days to get reward Default 300.00,
        n_simulations (int): Number of simulations to run in the Monte-Carlo Simulation

    Methods:
        _sliding_window:
            Utilizes a sliding window approach to calculate the amount of consecutive days a stock has been over the reward price
        simulation:
            Performs the Monte-Carlo simulation based on the initialized parameters
    """

    def __init__(
        self,
        stock_price: float,
        risk_free_rate: float,
        dividend_yield: float,
        volatility: float,
        reward_price: float,
        n_simulations: int,
    ):
        self.stock_price = stock_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.volatility = volatility
        self.reward_price = reward_price
        self.simulations = n_simulations
        self._time = 3
        self._trading_days = 252

    def _sliding_window(
        self, total_trading_days: int, stock_paths: np.array
    ) -> Tuple[np.array, np.array]:
        """Method Calculates the consecutive dates over the reward price by using a data masking approach.

        Parameters:
            total_trading_days(int): Total trading days in the preformance period.
            stock_paths(np.array): Stock paths generated utilizing a Brownian Motion formula assuming a risk-neutral framework

        Returns:
            final_stock_prices: The final stock prices of all of the vested stock paths.
            vested: Vector of the rows where the stock is above 300 for 20 consecutive days.
        """
        masked_stock_paths = stock_paths > self.reward_price
        # Setting the Window Size
        n_windows = total_trading_days - 20 + 1
        vested = np.zeros(self.simulations, dtype=bool)
        for i in range(n_windows):
            window = masked_stock_paths[:, i : i + 20]
            vested |= window.all(axis=1)
        final_stock_prices = stock_paths[vested, -1]
        return final_stock_prices, vested

    def _percentile_calculations(self, stock_paths: np.array) -> list:
        return np.percentile(stock_paths, [10, 25, 50, 75, 90], axis=0)

    def simulation(self) -> dict:
        """Method that performs the Monte Carlo simulation"""
        total_trading_days = int(self._time * self._trading_days)
        delta_t = 1 / self._trading_days
        generated_normal = np.random.standard_normal(
            (self.simulations, total_trading_days)
        )
        # Brownian Motion Formula
        S_t = np.exp(
            (self.risk_free_rate - self.dividend_yield - 0.5 * self.volatility**2)
            * delta_t
            + self.volatility * np.sqrt(delta_t) * generated_normal
        )
        stock_paths = self.stock_price * np.cumprod(S_t, axis=1)
        final_stock_price, vested = self._sliding_window(
            total_trading_days, stock_paths
        )
        payouts = final_stock_price * 100
        present_reward_values = payouts * np.exp(-self.risk_free_rate * self._time)
        present_reward_values.mean()
        fair_value = np.sum(present_reward_values) / self.simulations
        vest_pct = vested.mean() * 100
        percentiles = self._percentile_calculations(stock_paths=stock_paths)
        payload = {
            "stock_paths": stock_paths,
            "vested": vested,
            "total_trading_days": total_trading_days,
            "fair_value": fair_value,
            "vest_pct": vest_pct,
            "final_price_vested": stock_paths[vested, -1].mean(),
            "final_price_all_runs": stock_paths[:, -1].mean(),
            "percentiles": percentiles,
        }
        return payload
