from pathlib import Path

import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


class FutureForecaster:
    """
    Generate future forecasts using a trained LSTM model.
    """

    def __init__(self, model_path):

        self.model_path = Path(model_path)
        self.model = None

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------
    def load(self):

        self.model_path = Path(self.model_path).resolve()

        print("Loading:", self.model_path)
        print("Exists :", self.model_path.exists())

        self.model = load_model(str(self.model_path))

        print("Loaded successfully!")

        return self.model
    # --------------------------------------------------------
    # Multi-step forecasting
    # --------------------------------------------------------

    def forecast(
        self,
        last_sequence,
        scaler,
        future_days=252
    ):

        try:

            if self.model is None:

                raise ValueError(
                    "Model has not been loaded."
                )

            current_sequence = last_sequence.copy()

            predictions_scaled = []

            for _ in range(future_days):

                prediction = self.model.predict(
                    current_sequence,
                    verbose=0
                )

                predictions_scaled.append(
                    prediction[0, 0]
                )

                current_sequence = np.append(

                    current_sequence[:, 1:, :],

                    prediction.reshape(1, 1, 1),

                    axis=1

                )

            predictions_scaled = np.array(
                predictions_scaled
            ).reshape(-1, 1)

            predictions = scaler.inverse_transform(
                predictions_scaled
            )

            return predictions.flatten()

        except Exception as e:

            raise RuntimeError(
                f"Forecast failed.\n{e}"
            )

    # --------------------------------------------------------
    # Generate future dates
    # --------------------------------------------------------

    @staticmethod
    def create_future_dates(
        last_date,
        periods=252
    ):

        try:

            last_date = pd.to_datetime(last_date)

            future_dates = pd.bdate_range(

                start=last_date + pd.Timedelta(days=1),

                periods=periods

            )

            return future_dates

        except Exception as e:

            raise RuntimeError(
                f"Future date generation failed.\n{e}"
            )
    @staticmethod
    def confidence_intervals(predictions, daily_returns):

        """
        Approximate 95% confidence intervals using historical volatility.
        """

        try:

            volatility = daily_returns.std()

            lower = predictions * (1 - 1.96 * volatility)

            upper = predictions * (1 + 1.96 * volatility)

            return lower, upper

        except Exception as e:

            raise RuntimeError(
                f"Confidence interval calculation failed.\n{e}"
            )

    # --------------------------------------------------------
    # Trend Analysis
    # --------------------------------------------------------

    @staticmethod
    def analyze_trend(predictions):

        try:

            start_price = predictions[0]
            end_price = predictions[-1]

            pct_change = (
                (end_price - start_price)
                / start_price
            ) * 100

            if pct_change > 5:

                trend = "Upward"

            elif pct_change < -5:

                trend = "Downward"

            else:

                trend = "Stable"

            return {

                "Trend": trend,

                "Start Price": round(float(start_price),2),

                "End Price": round(float(end_price),2),

                "Percent Change": round(float(pct_change),2)

            }

        except Exception as e:

            raise RuntimeError(
                f"Trend analysis failed.\n{e}"
            )

    # --------------------------------------------------------
    # Opportunities & Risks
    # --------------------------------------------------------

    @staticmethod
    def market_analysis(trend):

        try:

            opportunities = []
            risks = []

            if trend["Trend"] == "Upward":

                opportunities.extend([

                    "Potential long-term capital appreciation.",

                    "Positive market momentum may support investment.",

                    "Suitable for long-term investors."

                ])

                risks.extend([

                    "Price corrections may occur after strong gains.",

                    "Forecast uncertainty increases over time."

                ])

            elif trend["Trend"] == "Downward":

                opportunities.extend([

                    "Potential buying opportunities after declines.",

                    "Useful for value investors."

                ])

                risks.extend([

                    "Expected price decline.",

                    "Higher downside risk.",

                    "Possible increased volatility."

                ])

            else:

                opportunities.extend([

                    "Stable market suitable for conservative investors.",

                    "Lower expected volatility."

                ])

                risks.extend([

                    "Limited capital appreciation.",

                    "Unexpected market events may change the trend."

                ])

            return opportunities, risks

        except Exception as e:

            raise RuntimeError(
                f"Market analysis failed.\n{e}"
            )

    # --------------------------------------------------------
    # Save Forecast
    # --------------------------------------------------------

    @staticmethod
    def save_forecast(
        dates,
        forecast,
        lower,
        upper,
        output_path
    ):

        try:

            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            df = pd.DataFrame({

                "Date": dates,

                "Forecast": forecast,

                "Lower_CI": lower,

                "Upper_CI": upper

            })

            df.to_csv(
                output_path,
                index=False
            )

            return df

        except Exception as e:

            raise RuntimeError(
                f"Saving forecast failed.\n{e}"
            )