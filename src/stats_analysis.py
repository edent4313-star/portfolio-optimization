from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

def summary_statistics(df):
    """Calculates stats for a single dataframe."""
    # Example logic (replace with your actual calculations)
    return df.describe() 

def summary_statistics_all(assets):
    """Loops through all assets and calculates stats."""
    results = {}
    for ticker, df in assets.items():
        # This is where it was failing because it couldn't find the function above
        results[ticker] = summary_statistics(df) 
    return results
def detect_outliers(df, column="Daily_Return"):
    """Detects outliers using the Interquartile Range (IQR) method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 - 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


def adf_test(series):
    """Performs the Augmented Dickey-Fuller test to check for stationarity."""
    # We drop NaNs because the test will fail if they are present
    result = adfuller(series.dropna())
    
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Stationary': result[1] < 0.05  # True if p-value is less than 5%
    }
def calculate_var(df, column="Daily_Return", confidence_level=0.95):
    """Calculates Value at Risk (VaR)."""
    return np.percentile(df[column].dropna(), (1 - confidence_level) * 100)

def calculate_sharpe_ratio(df, column="Daily_Return", risk_free_rate=0.0):
    """Calculates the Sharpe Ratio."""
    mean_return = df[column].mean()
    std_dev = df[column].std()
    return (mean_return - risk_free_rate) / std_dev
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


def adf_test(series):
    """
    Perform Augmented Dickey-Fuller (ADF) Test.

    Parameters
    ----------
    series : pandas.Series
        Time series data.

    Returns
    -------
    dict
        Dictionary containing ADF test results.
    """

    result = adfuller(series.dropna())

    output = {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Lags Used": result[2],
        "Number of Observations": result[3],
        "Critical Values": result[4],
    }

    print("=" * 60)
    print(f"ADF Statistic : {output['ADF Statistic']:.6f}")
    print(f"p-value       : {output['p-value']:.6f}")
    print(f"Lags Used     : {output['Lags Used']}")
    print(f"Observations  : {output['Number of Observations']}")
    print("\nCritical Values")

    for key, value in output["Critical Values"].items():
        print(f"{key}: {value:.4f}")

    if output["p-value"] < 0.05:
        print("\nConclusion:")
        print("The series is Stationary (Reject H0).")
    else:
        print("\nConclusion:")
        print("The series is Non-Stationary (Fail to Reject H0).")

    return output