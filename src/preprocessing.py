from pathlib import Path
from src.config import PROCESSED_DATA_DIR
from src.logger import logger
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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



class TimeSeriesPreprocessor:

    def __init__(self):
        self.scalers = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date")

            df = df.dropna(subset=["Date"])

            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].ffill().bfill()

            return df

        except Exception as e:
            raise RuntimeError(f"Preprocessing error: {e}")

    def train_test_split(self, df: pd.DataFrame, train_ratio=0.8):

        try:
            split = int(len(df) * train_ratio)

            train = df.iloc[:split].copy()
            test = df.iloc[split:].copy()

            return train, test

        except Exception as e:
            raise RuntimeError(f"Split error: {e}")

    def scale_close(self, train, test, col="Close"):

        try:
            scaler = MinMaxScaler()

            train_scaled = scaler.fit_transform(train[[col]])
            test_scaled = scaler.transform(test[[col]])

            self.scalers[col] = scaler

            return train_scaled, test_scaled, scaler

        except Exception as e:
            raise RuntimeError(f"Scaling error: {e}")

    def create_sequences(self, data, window=60):

        try:
            X, y = [], []

            for i in range(window, len(data)):
                X.append(data[i-window:i])
                y.append(data[i])

            return np.array(X), np.array(y)

        except Exception as e:
            raise RuntimeError(f"Sequence error: {e}")

def clean_data(self, df):

    try:

        df = df.copy()

        df = df.sort_values("Date")

        numeric_cols = df.select_dtypes(include="number").columns

        df[numeric_cols] = (
            df[numeric_cols]
            .ffill()
            .bfill()
        )

        return df

    except Exception as e:
        raise RuntimeError(f"Preprocessing error: {e}")