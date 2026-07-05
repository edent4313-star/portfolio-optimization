import yfinance as yf
import pandas as pd
from pathlib import Path

from src.logger import logger
from src.config import (
    RAW_DATA_DIR,
    START_DATE,
    END_DATE,
    TICKERS
)
from src.exceptions import (
    DataDownloadError,
    EmptyDataError
)


def download_asset(
    ticker,
    start_date,
    end_date
):
    try:

        logger.info(f"Downloading {ticker}")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False
        )

        if df.empty:
            raise EmptyDataError(
                f"No data returned for {ticker}"
            )

        output_file = RAW_DATA_DIR / f"{ticker}.csv"

        df.to_csv(output_file)

        logger.info(
            f"{ticker} saved successfully"
        )

        return df

    except Exception as e:

        logger.error(str(e))

        raise DataDownloadError(
            f"Failed downloading {ticker}"
        ) from e
    
def download_all_assets():
    """
    Download all configured assets.

    Returns
    -------
    dict
        Dictionary containing one DataFrame per ticker.
    """

    assets = {}

    for ticker in TICKERS:

        logger.info(f"Processing {ticker}")

        assets[ticker] = download_asset(
            ticker=ticker,
            start_date=START_DATE,
            end_date=END_DATE
        )

    logger.info("All assets downloaded successfully.")

    return assets

class DataLoader:

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):

        try:

            path = Path(self.file_path)

            if not path.exists():
                raise FileNotFoundError(f"{path} not found")

            df = pd.read_csv(path)

            # ---------------------------------
            # Remove the extra Yahoo Finance rows
            # ---------------------------------
            if str(df.iloc[0, 0]) == "Ticker":
                df = df.iloc[2:].reset_index(drop=True)

            # ---------------------------------
            # Rename first column to Date
            # ---------------------------------
            df.rename(
                columns={df.columns[0]: "Date"},
                inplace=True
            )

            # ---------------------------------
            # Convert date
            # ---------------------------------
            df["Date"] = pd.to_datetime(df["Date"])

            # ---------------------------------
            # Convert remaining columns to numeric
            # ---------------------------------
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            raise RuntimeError(f"Loading error: {e}")