from pathlib import Path
from pypfopt import EfficientFrontier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PortfolioOptimizer:
    """
    Portfolio Optimization using Modern Portfolio Theory.
    """

    def __init__(self,
                 forecast_df,
                 historical_returns):

        """
        Parameters
        ----------
        forecast_df : DataFrame
            Forecast generated for TSLA.

        historical_returns : DataFrame
            Daily return dataframe for
            TSLA, SPY and BND.
        """

        self.forecast_df = forecast_df
        self.returns = historical_returns

        self.expected_returns = None
        self.covariance_matrix = None

    # --------------------------------------------------------
    # Validate Input
    # --------------------------------------------------------

    def validate_data(self):

        try:

            if self.forecast_df.empty:
                raise ValueError(
                    "Forecast dataframe is empty."
                )

            if self.returns.empty:
                raise ValueError(
                    "Historical return dataframe is empty."
                )

            required = ["TSLA", "SPY", "BND"]

            for col in required:

                if col not in self.returns.columns:

                    raise ValueError(
                        f"{col} not found in return dataframe."
                    )

            if "Forecast" not in self.forecast_df.columns:

                raise ValueError(
                    "Forecast column missing."
                )

            return True

        except Exception as e:

            raise RuntimeError(
                f"Validation failed.\n{e}"
            )

    # --------------------------------------------------------
    # Prepare Expected Returns
    # --------------------------------------------------------

    def prepare_expected_returns(self):

        """
        Assignment requirement

        TSLA:
            Forecast return

        SPY:
            Historical annual return

        BND:
            Historical annual return
        """

        try:

            self.validate_data()

            expected = {}

            # -------------------------------
            # Forecast return (TSLA)
            # -------------------------------

            first_price = self.forecast_df[
                "Forecast"
            ].iloc[0]

            last_price = self.forecast_df[
                "Forecast"
            ].iloc[-1]

            tsla_return = (

                (last_price - first_price)

                / first_price

            )

            expected["TSLA"] = tsla_return

            # -------------------------------
            # Historical annual return
            # -------------------------------

            expected["SPY"] = (

                self.returns["SPY"].mean()

                * 252

            )

            expected["BND"] = (

                self.returns["BND"].mean()

                * 252

            )

            self.expected_returns = pd.Series(
                expected
            )

            return self.expected_returns

        except Exception as e:

            raise RuntimeError(
                f"Expected return calculation failed.\n{e}"
            )

    # --------------------------------------------------------
    # Covariance Matrix
    # --------------------------------------------------------

    def calculate_covariance(self):

        """
        Annual covariance matrix.
        """

        try:

            self.validate_data()

            self.covariance_matrix = (

                self.returns.cov()

                * 252

            )

            return self.covariance_matrix

        except Exception as e:

            raise RuntimeError(
                f"Covariance calculation failed.\n{e}"
            )

    # --------------------------------------------------------
    # Save Covariance Matrix
    # --------------------------------------------------------

    @staticmethod
    def save_covariance(
        covariance,
        output_path
    ):

        try:

            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            covariance.to_csv(
                output_path
            )

            print(
                f"Saved covariance matrix:\n{output_path}"
            )

        except Exception as e:

            raise RuntimeError(
                f"Saving covariance failed.\n{e}"
            )
    # --------------------------------------------------------
    # Maximum Sharpe Portfolio
    # --------------------------------------------------------

    def optimize_max_sharpe(self, risk_free_rate=0.02):

        """
        Optimize portfolio to maximize Sharpe Ratio.
        """

        try:

            if self.expected_returns is None:
                self.prepare_expected_returns()

            if self.covariance_matrix is None:
                self.calculate_covariance()

            ef = EfficientFrontier(
                self.expected_returns,
                self.covariance_matrix
            )

            ef.max_sharpe(
                risk_free_rate=risk_free_rate
            )

            weights = ef.clean_weights()

            performance = ef.portfolio_performance(
                risk_free_rate=risk_free_rate
            )

            return {

                "weights": weights,

                "performance": {

                    "Expected Return": performance[0],

                    "Volatility": performance[1],

                    "Sharpe Ratio": performance[2]

                }

            }

        except Exception as e:

            raise RuntimeError(
                f"Maximum Sharpe optimization failed.\n{e}"
            )
    # --------------------------------------------------------
    # Minimum Volatility Portfolio
    # --------------------------------------------------------

    def optimize_min_volatility(self):

        try:

            if self.expected_returns is None:
                self.prepare_expected_returns()

            if self.covariance_matrix is None:
                self.calculate_covariance()

            ef = EfficientFrontier(
                self.expected_returns,
                self.covariance_matrix
            )

            ef.min_volatility()

            weights = ef.clean_weights()

            performance = ef.portfolio_performance()

            return {

                "weights": weights,

                "performance": {

                    "Expected Return": performance[0],

                    "Volatility": performance[1],

                    "Sharpe Ratio": performance[2]

                }

            }

        except Exception as e:

            raise RuntimeError(
                f"Minimum volatility optimization failed.\n{e}"
            )
    # --------------------------------------------------------
    # Portfolio Metrics
    # --------------------------------------------------------

    @staticmethod
    def portfolio_metrics(weights,
                          expected_returns,
                          covariance_matrix,
                          risk_free_rate=0.02):

        """
        Calculate portfolio metrics manually.
        """

        try:

            weights = np.array(
                list(weights.values())
            )

            expected = np.array(
                expected_returns.values
            )

            annual_return = np.dot(
                weights,
                expected
            )

            volatility = np.sqrt(

                np.dot(

                    weights.T,

                    np.dot(
                        covariance_matrix.values,
                        weights
                    )

                )

            )

            sharpe = (

                annual_return - risk_free_rate

            ) / volatility

            return {

                "Expected Return": annual_return,

                "Volatility": volatility,

                "Sharpe Ratio": sharpe

            }

        except Exception as e:

            raise RuntimeError(
                f"Portfolio metric calculation failed.\n{e}"
            )
    # --------------------------------------------------------
# Efficient Frontier Simulation
# --------------------------------------------------------

    def generate_efficient_frontier(
        self,
        n_portfolios=5000,
        risk_free_rate=0.02,
        random_state=42
        ):
        """
        Generate random portfolios for Efficient Frontier.
        """

        try:

            if self.expected_returns is None:
                self.prepare_expected_returns()

            if self.covariance_matrix is None:
                self.calculate_covariance()

            import numpy as np
            import pandas as pd

            np.random.seed(random_state)

            num_assets = len(self.expected_returns)

            portfolio_returns = []
            portfolio_risk = []
            portfolio_sharpe = []
            portfolio_weights = []

            for _ in range(n_portfolios):

                weights = np.random.random(num_assets)
                weights /= np.sum(weights)

                annual_return = np.dot(
                weights,
                self.expected_returns.values
                )

                annual_risk = np.sqrt(
                    np.dot(
                    weights.T,
                    np.dot(
                        self.covariance_matrix.values,
                        weights
                    )
                )
            )

                sharpe = (
                annual_return - risk_free_rate
                 ) / annual_risk

                portfolio_returns.append(annual_return)
                portfolio_risk.append(annual_risk)
                portfolio_sharpe.append(sharpe)
                portfolio_weights.append(weights)

                frontier = pd.DataFrame({

                "Return": portfolio_returns,

                "Risk": portfolio_risk,

                "Sharpe": portfolio_sharpe,

                "Weights": portfolio_weights

                })
            return frontier

        except Exception as e:

            raise RuntimeError(
            f"Efficient Frontier generation failed.\n{e}"
        )
    # --------------------------------------------------------
# Save Portfolio Weights
# --------------------------------------------------------

    @staticmethod
    def save_portfolio(weights, output_file):

        try:

            output_file = Path(output_file)

            output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

            portfolio = pd.DataFrame({

            "Asset": list(weights.keys()),

            "Weight": list(weights.values())

        })

            portfolio.to_csv(
            output_file,
            index=False
        )

            print(
            f"Portfolio saved:\n{output_file}"
        )

            return portfolio

        except Exception as e:

            raise RuntimeError(
                f"Saving portfolio failed.\n{e}"
        )
# --------------------------------------------------------
# Save Portfolio Summary
# --------------------------------------------------------

    @staticmethod
    def save_summary(summary, output_file):

            try:

                output_file = Path(output_file)

                output_file.parent.mkdir(
                parents=True,
                exist_ok=True
                 )

                summary.to_csv(
                output_file,
                index=False
                 )

                print(
                f"Summary saved:\n{output_file}"
                )

            except Exception as e:

                raise RuntimeError(
            f"Saving summary failed.\n{e}"
        )
    # --------------------------------------------------------
# Summary DataFrame
# --------------------------------------------------------

    @staticmethod
    def summary_dataframe(

            max_sharpe,

             min_volatility

            ):

            try:

                    summary = pd.DataFrame({

                    "Portfolio":[

                    "Maximum Sharpe",

                    "Minimum Volatility"

                    ],

                    "Expected Return":[

                    max_sharpe["performance"]["Expected Return"],

                    min_volatility["performance"]["Expected Return"]

                    ],

                    "Volatility":[

                    max_sharpe["performance"]["Volatility"],

                    min_volatility["performance"]["Volatility"]

                    ],

                    "Sharpe Ratio":[

                    max_sharpe["performance"]["Sharpe Ratio"],

                    min_volatility["performance"]["Sharpe Ratio"]

                    ]

                    })

                    return summary

            except Exception as e:

                raise RuntimeError(
                f"Summary creation failed.\n{e}"
                )