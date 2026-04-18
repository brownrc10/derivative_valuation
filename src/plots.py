import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class PlotsGenerator:
    def __init__(self, simulation_payload: dict, barrier: float, strike: float):
        self.payload = simulation_payload
        self.barrier = barrier
        self.strike = strike

    def plot_final_price_distribution(self):
        stock_paths = self.payload["stock_paths"]
        knocked_in = self.payload["knocked_in"]

        all_final = stock_paths[:, -1]
        knocked_in_final = stock_paths[knocked_in, -1]
        not_knocked_in_final = stock_paths[~knocked_in, -1]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(
            all_final, ax=ax, label="All Paths", color="steelblue", alpha=0.4, bins=100
        )
        sns.histplot(
            knocked_in_final,
            ax=ax,
            label="Knocked-In",
            color="green",
            alpha=0.4,
            bins=100,
        )
        sns.histplot(
            not_knocked_in_final,
            ax=ax,
            label="Not Knocked-In",
            color="red",
            alpha=0.4,
            bins=100,
        )

        ax.axvline(
            self.barrier,
            color="orange",
            linestyle="--",
            label=f"Barrier ${self.barrier:.2f}",
        )
        ax.axvline(
            self.strike,
            color="purple",
            linestyle="--",
            label=f"Strike ${self.strike:.2f}",
        )

        ax.set_title("Distribution of Final Stock Prices")
        ax.set_xlabel("Final Stock Price")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_payoff_distribution(self):
        stock_paths = self.payload["stock_paths"]
        knocked_in = self.payload["knocked_in"]

        final_prices = stock_paths[knocked_in, -1]
        payoffs = np.maximum(final_prices - self.strike, 0)
        payoffs_discounted = self.payload["fair_value"]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(payoffs, ax=ax, color="green", alpha=0.6, bins=100)

        ax.axvline(
            payoffs.mean(),
            color="red",
            linestyle="--",
            label=f"Mean Payoff ${payoffs.mean():.2f}",
        )
        ax.axvline(
            np.median(payoffs),
            color="orange",
            linestyle="--",
            label=f"Median Payoff ${np.median(payoffs):.2f}",
        )

        ax.set_title("Distribution of Option Payoffs (Knocked-In Paths Only)")
        ax.set_xlabel("Payoff")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        plt.show()
