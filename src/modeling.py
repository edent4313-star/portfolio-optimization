from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR
)

from src.logger import logger
from src.exceptions import EmptyDataError


def load_processed_data(ticker: str) -> pd.DataFrame:
    """
    Load processed dataset.

    Parameters
    ----------
    ticker : str
        Asset ticker.

    Returns
    -------
    pandas.DataFrame
    """

    try:

        file_path = PROCESSED_DATA_DIR / f"{ticker}_processed.csv"

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_csv(file_path)

        if df.empty:
            raise EmptyDataError(
                f"{ticker} processed dataset is empty."
            )

        logger.info(f"{ticker} loaded successfully.")

        return df

    except Exception as e:

        logger.exception(e)
        raise


def split_train_test(
    df: pd.DataFrame,
    train_end: str = "2024-12-31"
):
    """
    Chronological split.

    Returns
    -------
    train_df
    test_df
    """

    train = df.loc[:train_end].copy()

    test = df.loc[train_end:].copy()

    if len(train) == 0:
        raise ValueError("Training dataset is empty.")

    if len(test) == 0:
        raise ValueError("Testing dataset is empty.")

    logger.info(
        f"Train={len(train)} Test={len(test)}"
    )

    return train, test


def scale_close_price(
    train_df,
    test_df
):
    """
    Scale Close column for LSTM.

    Returns
    -------
    train_scaled
    test_scaled
    scaler
    """

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_df[["Close"]]
    )

    test_scaled = scaler.transform(
        test_df[["Close"]]
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.pkl"
    )

    logger.info("Scaler saved.")

    return (
        train_scaled,
        test_scaled,
        scaler
    )


def load_scaler():
    """
    Load saved scaler.
    """

    scaler = joblib.load(
        MODELS_DIR / "scaler.pkl"
    )

    return scaler


def create_sequences(
    data,
    window_size=60
):
    """
    Create LSTM sequences.

    Parameters
    ----------
    data : ndarray

    window_size : int

    Returns
    -------
    X
    y
    """

    X = []

    y = []

    for i in range(
        window_size,
        len(data)
    ):

        X.append(
            data[
                i-window_size:i,
                0
            ]
        )

        y.append(
            data[
                i,
                0
            ]
        )

    X = np.array(X)

    y = np.array(y)

    X = X.reshape(
        (
            X.shape[0],
            X.shape[1],
            1
        )
    )

    logger.info(
        f"Created {len(X)} sequences."
    )

    return X, y


def inverse_transform(
    scaler,
    values
):
    """
    Convert scaled predictions back to price.
    """

    values = values.reshape(-1, 1)

    return scaler.inverse_transform(values)