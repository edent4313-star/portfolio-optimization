import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

def plot_closing_prices(assets):

    plt.figure(figsize=(14,6))

    for ticker, df in assets.items():

        plt.plot(
            df.index,
            df["Close"],
            label=ticker
        )

    plt.legend()

    plt.show()

def plot_close_price(df, ticker):
    """
    Plot Closing Price.
    """

    plt.figure(figsize=(12,6))

    plt.plot(
        df.index,
        df["Close"],
        color="blue",
        linewidth=2
    )

    plt.title(f"{ticker} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")

    plt.grid(True)

    plt.show()

def plot_daily_return(df, ticker):
    """
    Plot Daily Return.
    """

    plt.figure(figsize=(12,6))

    plt.plot(
        df.index,
        df["Daily_Return"],
        color="green"
    )

    plt.title(f"{ticker} Daily Return")

    plt.xlabel("Date")

    plt.ylabel("Return")

    plt.grid(True)

    plt.show()
class Visualizer:

    @staticmethod
    def plot_forecast(actual, predicted, title):

        try:
            plt.figure(figsize=(12,5))

            plt.plot(actual, label="Actual")
            plt.plot(predicted, label="Predicted")

            plt.title(title)
            plt.legend()
            plt.grid(True)

            plt.show()

        except Exception as e:
            raise RuntimeError(f"Plot error: {e}")

    @staticmethod
    def plot_future_forecast(
        history_dates,
        history_prices,
        future_dates,
        forecast,
        lower,
        upper,
        title="Future Stock Forecast"
    ):

        try:

            plt.figure(figsize=(16,6))

            plt.plot(

                history_dates,

                history_prices,

                label="Historical",

                linewidth=2

            )

            plt.plot(

                future_dates,

                forecast,

                color="red",

                linewidth=2,

                label="Forecast"

            )

            plt.fill_between(

                future_dates,

                lower,

                upper,

                color="red",

                alpha=0.2,

                label="95% Confidence Interval"

            )

            plt.title(title)

            plt.xlabel("Date")

            plt.ylabel("Close Price")

            plt.legend()

            plt.grid(True)

            plt.tight_layout()

            plt.show()

        except Exception as e:

            raise RuntimeError(
                f"Visualization failed.\n{e}"
            )
class PortfolioVisualizer:
    """
    Visualization utilities for portfolio optimization.
    """

    def __init__(self):

        plt.style.use("ggplot")
    # --------------------------------------------------------
    # Covariance Heatmap
    # --------------------------------------------------------

    @staticmethod
    def covariance_heatmap(
        covariance_matrix,
        save_path=None
    ):

        try:

            plt.figure(figsize=(8,6))

            plt.imshow(
                covariance_matrix,
                interpolation="nearest"
            )

            plt.colorbar(label="Covariance")

            plt.xticks(
                range(len(covariance_matrix.columns)),
                covariance_matrix.columns,
                rotation=45
            )

            plt.yticks(
                range(len(covariance_matrix.columns)),
                covariance_matrix.columns
            )

            for i in range(len(covariance_matrix)):
                for j in range(len(covariance_matrix.columns)):

                    plt.text(
                        j,
                        i,
                        f"{covariance_matrix.iloc[i,j]:.4f}",
                        ha="center",
                        va="center",
                        fontsize=9
                    )

            plt.title("Annual Covariance Matrix")

            plt.tight_layout()

            if save_path:

                save_path = Path(save_path)

                save_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                plt.savefig(
                    save_path,
                    dpi=300
                )

            plt.show()

        except Exception as e:

            raise RuntimeError(
                f"Unable to plot covariance matrix.\n{e}"
            )
    # --------------------------------------------------------
    # Efficient Frontier
    # --------------------------------------------------------

    @staticmethod
    def efficient_frontier(
        frontier,
        max_sharpe,
        min_volatility,
        save_path=None
    ):

        try:

            plt.figure(figsize=(10,7))

            plt.scatter(

                frontier["Risk"],

                frontier["Return"],

                c=frontier["Sharpe"],

                cmap="viridis",

                alpha=0.6

            )

            plt.colorbar(
                label="Sharpe Ratio"
            )

            plt.scatter(

                max_sharpe["performance"]["Volatility"],

                max_sharpe["performance"]["Expected Return"],

                marker="*",

                s=350,

                color="red",

                label="Maximum Sharpe"

            )

            plt.scatter(

                min_volatility["performance"]["Volatility"],

                min_volatility["performance"]["Expected Return"],

                marker="X",

                s=220,

                color="blue",

                label="Minimum Volatility"

            )

            plt.xlabel(
                "Annual Volatility"
            )

            plt.ylabel(
                "Expected Annual Return"
            )

            plt.title(
                "Efficient Frontier"
            )

            plt.legend()

            plt.grid(True)

            if save_path:

                save_path = Path(save_path)

                save_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                plt.savefig(
                    save_path,
                    dpi=300
                )

            plt.show()

        except Exception as e:

            raise RuntimeError(
                f"Unable to plot Efficient Frontier.\n{e}"
            )
    # --------------------------------------------------------
    # Portfolio Allocation
    # --------------------------------------------------------

    @staticmethod
    def portfolio_allocation(
        weights,
        save_path=None
    ):

        try:

            labels = list(weights.keys())

            values = list(weights.values())

            plt.figure(figsize=(7,7))

            plt.pie(

                values,

                labels=labels,

                autopct="%1.1f%%",

                startangle=90

            )

            plt.title(
                "Optimal Portfolio Allocation"
            )

            if save_path:

                save_path = Path(save_path)

                save_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                plt.savefig(
                    save_path,
                    dpi=300
                )

            plt.show()

        except Exception as e:

            raise RuntimeError(
                f"Unable to plot portfolio allocation.\n{e}"
            )