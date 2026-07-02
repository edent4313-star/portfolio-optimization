import numpy as np

def calculate_var(df, confidence_level=0.95):
    """Calculates Value at Risk (VaR)."""
    # Assuming 'Daily_Return' column exists
    returns = df['Daily_Return'].dropna()
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_sharpe_ratio(df, risk_free_rate=0.0):
    """Calculates the Sharpe Ratio."""
    returns = df['Daily_Return'].dropna()
    mean_return = returns.mean() * 252 # Annualize
    std_dev = returns.std() * np.sqrt(252) # Annualize
    return (mean_return - risk_free_rate) / std_dev

def calculate_metrics_all(assets):
    """Loops through all assets to calculate risk metrics."""
    metrics = {}
    for ticker, df in assets.items():
        # These will work now because they are defined above
        metrics[ticker] = {
            "VaR": calculate_var(df),
            "Sharpe": calculate_sharpe_ratio(df)
        }
    return metrics