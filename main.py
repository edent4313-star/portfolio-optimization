'''from src.data_loader import download_asset
from src.config import (
    START_DATE,
    END_DATE,
    TICKERS
)

for ticker in TICKERS:

    download_asset(
        ticker,
        START_DATE,
        END_DATE
    )

print("Download Completed")'''


from src.data_loader import download_all_assets

assets = download_all_assets()

print("Download completed.")

print(assets.keys())