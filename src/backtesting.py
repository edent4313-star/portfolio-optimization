from pathlib import Path

import numpy as np
import pandas as pd


class Backtester:

    """
    Backtest optimized portfolio against benchmark.
    """

    def __init__(
        self,
        returns,
        portfolio_weights,
        benchmark_weights=None
    ):

        self.returns = returns.copy()

        self.portfolio_weights = portfolio_weights

        if benchmark_weights is None:

            benchmark_weights = {

                "SPY": 0.60,

                "BND": 0.40,

                "TSLA": 0.00

            }

        self.benchmark_weights = benchmark_weights

    # -------------------------------------------------------
    # Validate
    # -------------------------------------------------------

    def validate_data(self):

        try:

            if self.returns.empty:

                raise ValueError(
                    "Return dataframe is empty."
                )

            required = [

                "TSLA",

                "SPY",

                "BND"

            ]

            for col in required:

                if col not in self.returns.columns:

                    raise ValueError(
                        f"{col} missing."
                    )

            return True

        except Exception as e:

            raise RuntimeError(
                f"Validation failed.\n{e}"
            )

    # -------------------------------------------------------
    # Prepare Backtesting Window
    # -------------------------------------------------------

    def prepare_backtest_data(
        self,
        periods=252
    ):

        """
        Last one year.
        """

        try:

            self.validate_data()

            backtest = self.returns.tail(
                periods
            )

            return backtest

        except Exception as e:

            raise RuntimeError(
                f"Unable to prepare backtest.\n{e}"
            )

    # -------------------------------------------------------
    # Benchmark Portfolio
    # -------------------------------------------------------

    def benchmark_returns(
        self,
        backtest_returns
    ):

        try:

            weights = np.array([

                self.benchmark_weights["TSLA"],

                self.benchmark_weights["SPY"],

                self.benchmark_weights["BND"]

            ])

            benchmark = backtest_returns.dot(
                weights
            )

            return benchmark

        except Exception as e:

            raise RuntimeError(
                f"Benchmark calculation failed.\n{e}"
            )

    # -------------------------------------------------------
    # Strategy Returns
    # -------------------------------------------------------

    def strategy_returns(
        self,
        backtest_returns
    ):

        """
        Buy-and-hold strategy using
        optimal weights.
        """

        try:

            weights = np.array([

                self.portfolio_weights["TSLA"],

                self.portfolio_weights["SPY"],

                self.portfolio_weights["BND"]

            ])

            strategy = backtest_returns.dot(
                weights
            )

            return strategy

        except Exception as e:

            raise RuntimeError(
                f"Strategy simulation failed.\n{e}"
            )
        # -------------------------------------------------------
    # Cumulative Returns
    # -------------------------------------------------------

    @staticmethod
    def cumulative_returns(daily_returns):
        """
        Calculate cumulative portfolio returns.
        """

        try:

            cumulative = (1 + daily_returns).cumprod()

            return cumulative

        except Exception as e:

            raise RuntimeError(
                f"Cumulative return calculation failed.\n{e}"
            )

    # -------------------------------------------------------
    # Total Return
    # -------------------------------------------------------

    @staticmethod
    def total_return(daily_returns):
        """
        Calculate total portfolio return.
        """

        try:

            total = (1 + daily_returns).prod() - 1

            return total

        except Exception as e:

            raise RuntimeError(
                f"Total return calculation failed.\n{e}"
            )

    # -------------------------------------------------------
    # Annualized Return
    # -------------------------------------------------------

    @staticmethod
    def annualized_return(
        daily_returns,
        trading_days=252
    ):
        """
        Calculate annualized return.
        """

        try:

            cumulative = (1 + daily_returns).prod()

            years = len(daily_returns) / trading_days

            annual_return = cumulative ** (1 / years) - 1

            return annual_return

        except Exception as e:

            raise RuntimeError(
                f"Annualized return calculation failed.\n{e}"
            )

    # -------------------------------------------------------
    # Annualized Volatility
    # -------------------------------------------------------

    @staticmethod
    def annualized_volatility(
        daily_returns,
        trading_days=252
    ):
        """
        Calculate annualized volatility.
        """

        try:

            volatility = daily_returns.std() * np.sqrt(trading_days)

            return volatility

        except Exception as e:

            raise RuntimeError(
                f"Annualized volatility calculation failed.\n{e}"
            )

    # -------------------------------------------------------
    # Sharpe Ratio
    # -------------------------------------------------------

    @staticmethod
    def sharpe_ratio(
        daily_returns,
        risk_free_rate=0.02,
        trading_days=252
    ):
        """
        Annualized Sharpe Ratio.
        """

        try:

            annual_return = Backtester.annualized_return(
                daily_returns,
                trading_days
            )

            annual_volatility = Backtester.annualized_volatility(
                daily_returns,
                trading_days
            )

            sharpe = (
                annual_return - risk_free_rate
            ) / annual_volatility

            return sharpe

        except Exception as e:

            raise RuntimeError(
                f"Sharpe Ratio calculation failed.\n{e}"
            )

    # -------------------------------------------------------
    # Maximum Drawdown
    # -------------------------------------------------------

    @staticmethod
    def maximum_drawdown(daily_returns):
        """
        Calculate maximum drawdown.
        """

        try:

            cumulative = (1 + daily_returns).cumprod()

            running_max = cumulative.cummax()

            drawdown = (

                cumulative

                / running_max

            ) - 1

            return drawdown.min()

        except Exception as e:

            raise RuntimeError(
                f"Maximum drawdown calculation failed.\n{e}"
            )

    # -------------------------------------------------------
    # Performance Summary
    # -------------------------------------------------------

    def performance_summary(
        self,
        strategy_returns,
        benchmark_returns
    ):
        """
        Create summary table for strategy and benchmark.
        """

        try:

            summary = pd.DataFrame({

                "Metric": [

                    "Total Return",

                    "Annualized Return",

                    "Annualized Volatility",

                    "Sharpe Ratio",

                    "Maximum Drawdown"

                ],

                "Strategy": [

                    self.total_return(strategy_returns),

                    self.annualized_return(strategy_returns),

                    self.annualized_volatility(strategy_returns),

                    self.sharpe_ratio(strategy_returns),

                    self.maximum_drawdown(strategy_returns)

                ],

                "Benchmark": [

                    self.total_return(benchmark_returns),

                    self.annualized_return(benchmark_returns),

                    self.annualized_volatility(benchmark_returns),

                    self.sharpe_ratio(benchmark_returns),

                    self.maximum_drawdown(benchmark_returns)

                ]

            })

            return summary

        except Exception as e:

            raise RuntimeError(
                f"Performance summary failed.\n{e}"
            )

    # -------------------------------------------------------
    # Save Results
    # -------------------------------------------------------

    @staticmethod
    def save_results(
        dataframe,
        output_file
    ):
        """
        Save DataFrame to CSV.
        """

        try:

            output_file = Path(output_file)

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            dataframe.to_csv(
                output_file,
                index=False
            )

            print(f"Saved: {output_file}")

        except Exception as e:

            raise RuntimeError(
                f"Unable to save results.\n{e}"
            )