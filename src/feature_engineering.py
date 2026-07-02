
def add_features(df, ticker):
    # This line is likely missing!
    df[("Daily_Return", ticker)] = df[("Close", ticker)].pct_change()
    
    # You can add other features here too
    # df[("MA20", ticker)] = df[("Close", ticker)].rolling(window=20).mean()
    
    return df

def add_features_all(assets):
    featured = {}
    for ticker, df in assets.items():
        # This calls the function above for every stock
        featured[ticker] = add_features(df.copy(), ticker) 
    return featured

from sklearn.preprocessing import MinMaxScaler


def add_features(df,ticker):
    """
    Add engineered features to the dataset.
    """

    df = df.copy()

    # Daily return
    df[("Daily_Return", ticker)] = df[("Close", ticker)].pct_change()

    # Rolling statistics
    df[("Rolling_Mean_20", ticker)] = df[("Close", ticker)].rolling(20).mean()
    df[("Rolling_STD_20", ticker)] = df[("Close", ticker)].rolling(20).std()

    # Scale Close for LSTM
    scaler = MinMaxScaler()

    df[("Close_Scaled", ticker)] = scaler.fit_transform(df[("Close", ticker)].values.reshape(-1, 1))

    return df