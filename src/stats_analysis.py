from statsmodels.tsa.stattools import adfuller

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