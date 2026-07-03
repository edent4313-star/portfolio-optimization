from pathlib import Path
from src.config import PROCESSED_DATA_DIR
from src.logger import logger

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
    



def save_processed_data(assets):
    """
    Save processed DataFrames as CSV files.

    Parameters
    ----------
    assets : dict
        Dictionary containing processed DataFrames.
    """

    try:

        for ticker, df in assets.items():

            output_file = PROCESSED_DATA_DIR / f"{ticker}_processed.csv"

            df.to_csv(output_file)

            logger.info(f"{ticker} processed data saved to {output_file}")

        print("All processed datasets have been saved successfully.")

    except Exception as e:

        logger.error(f"Error saving processed data: {e}")

        raise

