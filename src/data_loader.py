import yfinance as yf
import pandas as pd

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
