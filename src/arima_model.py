import joblib
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pmdarima import auto_arima

from src.logger import logger
from src.config import MODELS_DIR


def plot_acf_pacf(series, lags=40):
    """
    Plot ACF and PACF.
    """

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(
        series.dropna(),
        lags=lags,
        ax=ax[0]
    )

    ax[0].set_title("Autocorrelation Function (ACF)")

    plot_pacf(
        series.dropna(),
        lags=lags,
        ax=ax[1],
        method="ywm"
    )

    ax[1].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout()

    plt.show()


def find_best_arima(series):
    """
    Auto ARIMA model selection.
    """

    logger.info("Searching best ARIMA parameters...")

    model = auto_arima(
        series,
        seasonal=False,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True
    )

    logger.info(model.summary())

    return model


def train_arima(series, order):
    """
    Train ARIMA.
    """

    logger.info(f"Training ARIMA{order}")

    model = ARIMA(
        series,
        order=order
    )

    fitted_model = model.fit()

    logger.info("ARIMA training completed.")

    return fitted_model


def train_sarima(
    series,
    order,
    seasonal_order
):
    """
    Train SARIMA.
    """

    logger.info("Training SARIMA")

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit()

    logger.info("SARIMA training completed.")

    return fitted_model


def forecast_model(
    model,
    steps
):
    """
    Forecast future values.
    """

    forecast = model.get_forecast(
        steps=steps
    )

    prediction = forecast.predicted_mean

    confidence_interval = forecast.conf_int()

    return (
        prediction,
        confidence_interval
    )


def plot_forecast(
    train,
    test,
    prediction,
    title="Forecast"
):
    """
    Plot forecast results.
    """

    plt.figure(figsize=(14,6))

    plt.plot(
        train.index,
        train,
        label="Train"
    )

    plt.plot(
        test.index,
        test,
        label="Test"
    )

    plt.plot(
        prediction.index,
        prediction,
        label="Forecast"
    )

    plt.title(title)

    plt.xlabel("Date")

    plt.ylabel("Closing Price")

    plt.legend()

    plt.grid(True)

    plt.show()


def plot_forecast_ci(
    train,
    test,
    prediction,
    confidence_interval
):
    """
    Plot forecast with confidence interval.
    """

    plt.figure(figsize=(14,6))

    plt.plot(
        train.index,
        train,
        label="Train"
    )

    plt.plot(
        test.index,
        test,
        label="Actual"
    )

    plt.plot(
        prediction.index,
        prediction,
        label="Forecast"
    )

    plt.fill_between(
        confidence_interval.index,
        confidence_interval.iloc[:,0],
        confidence_interval.iloc[:,1],
        alpha=0.25,
        label="95% CI"
    )

    plt.legend()

    plt.grid(True)

    plt.title("ARIMA Forecast")

    plt.show()


def save_model(
    model,
    filename="arima.pkl"
):
    """
    Save trained model.
    """

    file_path = MODELS_DIR / filename

    joblib.dump(
        model,
        file_path
    )

    logger.info(
        f"Model saved to {file_path}"
    )


def load_model(
    filename="arima.pkl"
):
    """
    Load trained model.
    """

    file_path = MODELS_DIR / filename

    model = joblib.load(file_path)

    logger.info(
        f"Loaded {filename}"
    )

    return model