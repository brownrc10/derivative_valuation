import numpy as np
from typing import Tuple


class BarrierMonteCarloSimulation:
    """A class that preforms the MonteCarlo simulation

    Parameters:
        stock_price (float): The stock price at the opening of the Performance Period
        risk_free_rate (float): Risk Free Rate calculated using the 2 Year Treasury Yield
        dividend_yield (float): Dividend Yield of VOO
        volatility (float): Volatility Calculation Based of Historical Stock Price Data
        reward_price (float): 10% increase of current_stock position
        n_simulations (int): Number of simulations to run in the Monte-Carlo Simulation

    Methods:
        _barrier_check:
            Checks if each simulated path ever crosses the barrier
        simulation:
            Performs the Monte-Carlo simulation based on the initialized parameters
    """

    def __init__(
        self,
        stock_price: float,
        risk_free_rate: float,
        dividend_yield: float,
        volatility: float,
        barrier: float,
        strike: float,
        n_simulations: int,
    ):
        self.stock_price = stock_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.volatility = volatility
        self.barrier = barrier
        self.strike = strike
        self.simulations = n_simulations
        self._time = 2
        self._trading_days = 252

    def _barrier_check(self, stock_paths: np.array) -> Tuple[np.array, np.array]:
        knocked_in = np.any(stock_paths > self.barrier, axis=1)
        final_stock_prices_knocked_in = stock_paths[knocked_in, -1]
        return final_stock_prices_knocked_in, knocked_in

    def _percentile_calculations(self, stock_paths: np.array) -> list:
        return np.percentile(stock_paths, [10, 25, 50, 75, 90], axis=0)

    def simulation(self) -> dict:
        """Method that performs the Monte Carlo simulation"""
        total_trading_days = int(self._time * self._trading_days)
        delta_t = 1 / self._trading_days
        generated_normal = np.random.standard_normal(
            (self.simulations, total_trading_days)
        )
        # Brownian Motion Formula under risk-neutral measure
        S_t = np.exp(
            (self.risk_free_rate - self.dividend_yield - 0.5 * self.volatility**2)
            * delta_t
            + self.volatility * np.sqrt(delta_t) * generated_normal
        )
        stock_paths = self.stock_price * np.cumprod(S_t, axis=1)
        final_stock_price, knocked_in = self._barrier_check(stock_paths)
        payoffs = np.maximum(final_stock_price - self.strike, 0)
        present_reward_values = payoffs * np.exp(-self.risk_free_rate * self._time)
        fair_value = np.sum(present_reward_values) / self.simulations
        knocked_in_pct = knocked_in.mean() * 100
        percentiles = self._percentile_calculations(stock_paths=stock_paths)
        payload = {
            "stock_paths": stock_paths,
            "knocked_in": knocked_in,
            "total_trading_days": total_trading_days,
            "fair_value": fair_value,
            "knocked_in_pct": knocked_in_pct,
            "final_price_vested": stock_paths[knocked_in, -1].mean(),
            "final_price_all_runs": stock_paths[:, -1].mean(),
            "percentiles": percentiles,
        }
        return payload
