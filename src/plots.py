import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PlotsGenerator:
    def __init__(
        self, simulation_payload: dict, barrier: float, strike: float, label: str
    ):
        self.payload = simulation_payload
        self.barrier = barrier
        self.strike = strike
        self.label = label
        self.business_days = pd.bdate_range(
            start=datetime.date(2024, 4, 30),
            periods=simulation_payload["total_trading_days"],
        )

    def _plot_final_price_distribution(self):
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
        plt.savefig(f"{self.label}")
        # plt.show()

    def _plot_payoff_distribution(self):
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
        plt.savefig(f"pay_off-{self.label}")
        # plt.show()

    def _plot_sample_paths(self, n_paths: int = 200):
        stock_paths = self.payload["stock_paths"]
        knocked_in = self.payload["knocked_in"]
        total_days = self.payload["total_trading_days"]

        fig, ax = plt.subplots(figsize=(14, 7))

        for i in np.where(~knocked_in)[0][: n_paths // 2]:
            ax.plot(
                self.business_days,
                stock_paths[i],
                color="red",
                alpha=0.05,
                linewidth=0.5,
            )

        for i in np.where(knocked_in)[0][: n_paths // 2]:
            ax.plot(
                self.business_days,
                stock_paths[i],
                color="green",
                alpha=0.05,
                linewidth=0.5,
            )

        ax.axhline(
            self.barrier,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Barrier ${self.barrier:.2f}",
        )
        ax.axhline(
            self.strike,
            color="purple",
            linestyle="--",
            linewidth=1.5,
            label=f"Strike ${self.strike:.2f}",
        )

        ax.plot([], [], color="green", alpha=0.5, label="Knocked-In Paths")
        ax.plot([], [], color="red", alpha=0.5, label="Not Knocked-In Paths")

        ax.set_title("Simulated Stock Price Paths")
        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Stock Price")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _plot_percentile_fan(self):
        stock_paths = self.payload["stock_paths"]
        percentiles = self.payload["percentiles"]
        total_days = self.payload["total_trading_days"]

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.fill_between(
            self.business_days,
            percentiles[0],
            percentiles[4],
            alpha=0.15,
            color="steelblue",
            label="10th-90th",
        )
        ax.fill_between(
            self.business_days,
            percentiles[1],
            percentiles[3],
            alpha=0.25,
            color="steelblue",
            label="25th-75th",
        )
        ax.plot(
            self.business_days,
            percentiles[2],
            color="steelblue",
            linewidth=2,
            label="Median (50th)",
        )

        ax.axhline(
            self.barrier,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Barrier ${self.barrier:.2f}",
        )
        ax.axhline(
            self.strike,
            color="purple",
            linestyle="--",
            linewidth=1.5,
            label=f"Strike ${self.strike:.2f}",
        )

        ax.set_title("Simulated Stock Price Paths — Percentile Fan Chart")
        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Stock Price")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _plot_comparison(self, analytical_prices: dict, mc_prices: dict):
        labels = list(analytical_prices.keys())
        analytical_vals = list(analytical_prices.values())
        mc_vals = list(mc_prices.values())

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(
            x - width / 2,
            analytical_vals,
            width,
            label="Analytical BS",
            color="steelblue",
            alpha=0.7,
        )
        bars2 = ax.bar(
            x + width / 2, mc_vals, width, label="Monte Carlo", color="green", alpha=0.7
        )

        for bar in bars1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"${bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"${bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title("Analytical BS vs Monte Carlo Fair Value by Vol Estimate")
        ax.set_xlabel("Volatility Estimate")
        ax.set_ylabel("Option Fair Value ($)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def generate_plots(self):
        self._plot_final_price_distribution()
        self._plot_payoff_distribution()
        self._plot_sample_paths()
        self._plot_percentile_fan()
