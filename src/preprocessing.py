def clean_dataset(df):
    # Your logic to clean a single dataframe goes here
    df = df.dropna() 
    return df

def clean_all_data(assets):
    cleaned = {}
    for ticker, df in assets.items():
        # This line was failing because it couldn't find the function above
        cleaned[ticker] = clean_dataset(df) 
    return cleaned