import numpy as np
from scipy.stats import norm


class BarrierOption:
    """Class used to calculate the price of an Up-and-In barrier Option

    Parameters:
        stock_price: Initial Price of the stock
        risk_free_rate: Continous risk free rate of a 2-Year Treasury
        dividend_yield: Dividend Yield Rate
        volatility: Volatility of the stock
        barrier: Barrier for the option that determines whether it is valid
        strike: Strike Price
        time_years: Total time to exercise the option
    Methods:
        _delta
        _up_and_out_call
        _vanilla_call
        up_and_in_call
    Note:
        With the exception of the up_and_in_call, the rest of the methods are intermediate
        steps to calculate the price of an up and out call and the vanilla call.
        Closed form solution of the call proved during instruction
    """

    def __init__(
        self,
        stock_price: float,
        risk_free_rate: float,
        dividend_yield: float,
        volatility: float,
        barrier: float,
        strike: float,
        time_years: int = 2,
    ):
        self.stock_price = stock_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.volatility = volatility
        self.barrier = barrier
        self.strike = strike
        self._time = time_years

    def _delta(self, x, sign):
        return (
            np.log(x)
            + (
                self.risk_free_rate
                - self.dividend_yield
                + sign * 0.5 * self.volatility**2
            )
            * self._time
        ) / (self.volatility * np.sqrt(self._time))

    def _up_and_out_call(self):
        S = self.stock_price
        K = self.strike
        b = self.barrier
        r = self.risk_free_rate
        v = self.volatility
        T = self._time

        power1 = (2 * (self.risk_free_rate - self.dividend_yield)) / self.volatility**2
        power2 = (
            1 - (2 * (self.risk_free_rate - self.dividend_yield)) / self.volatility**2
        )
        term_1 = S * (norm.cdf(self._delta(S / K, 1)) - norm.cdf(self._delta(S / b, 1)))

        term_2 = (
            b
            * (b / S) ** power1
            * (
                norm.cdf(self._delta(b**2 / (K * S), 1))
                - norm.cdf(self._delta(b / S, 1))
            )
        )
        term_3 = (
            K
            * np.exp(-r * T)
            * (norm.cdf(self._delta(S / K, -1)) - norm.cdf(self._delta(S / b, -1)))
        )
        term_4 = (
            K
            * np.exp(-r * T)
            * (S / b) ** power2
            * (
                norm.cdf(self._delta(b**2 / (K * S), -1))
                - norm.cdf(self._delta(b / S, -1))
            )
        )
        return term_1 - term_2 - term_3 + term_4

    def _vanilla_call(self):
        S = self.stock_price
        K = self.strike
        r = self.risk_free_rate
        q = self.dividend_yield
        v = self.volatility
        T = self._time

        d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def up_and_in_call(self):
        vanilla_call = self._vanilla_call()
        up_and_out = self._up_and_out_call()
        # print(f"Vanilla {vanilla_call}, Up_and_Out {up_and_out}")
        return vanilla_call - up_and_out
