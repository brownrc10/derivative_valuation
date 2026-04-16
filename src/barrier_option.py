import numpy as np
from scipy.stats import norm


class BarrierOption:
    def __init__(
        self,
        stock_price: float,
        risk_free_rate: float,
        dividend_yield: float,
        volatility: float,
        barrier: float,
        strike: float,
    ):
        self.stock_price = stock_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.volatility = volatility
        self.barrier = barrier
        self.strike = strike
        self.Time = 3

    @property
    def _scholes_lambda(self):
        return (
            self.risk_free_rate
            - self.dividend_yield
            + 0.5 * self.volatility**2 / self.volatility**2
        )

    @property
    def _scholes_y(self):
        return (
            np.log(self.barrier**2 / (self.strike * self.strike))
            / (self.volatility * np.sqrt(self.Time))
        ) + self._scholes_lambda * self.volatility * np.sqrt(self.Time)

    def barrier_call(self):
        term_1 = (
            self.stock_price
            * np.ex(-self.dividend_yield * self.Time)
            * (self.barrier / self.stock_price) ** (2 * self._scholes_lambda)
            * norm.cdf(self._scholes_y)
        )
        term_2 = (
            self.strike
            * np.exp(-self.risk_free_rate * self.Time)
            * (self.barrier / self.stock_price) ** (2 * self._scholes_lambda - 2)
            * norm.cdf(self._scholes_y - self.volatility - np.sqrt(self.Time))
        )
        return term_1 - term_2
