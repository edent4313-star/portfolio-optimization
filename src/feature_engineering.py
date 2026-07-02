
def add_features(df):
    # This line is likely missing!
    df['Daily_Return'] = df['Close'].pct_change()
    
    # You can add other features here too
    # df['MA20'] = df['Close'].rolling(window=20).mean()
    
    return df

def add_features_all(assets):
    featured = {}
    for ticker, df in assets.items():
        # This calls the function above for every stock
        featured[ticker] = add_features(df.copy()) 
    return featured