from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures" / "eda"
LOG_DIR = BASE_DIR / "logs"

# Create directories
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Project Configuration
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"

TICKERS = [
    "TSLA",
    "BND",
    "SPY"
]