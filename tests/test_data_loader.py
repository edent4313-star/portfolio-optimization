from src.data_loader import download_all_assets


def test_download_all_assets():

    assets = download_all_assets()

    assert "TSLA" in assets
    assert "BND" in assets
    assert "SPY" in assets

    assert not assets["TSLA"].empty
    assert not assets["BND"].empty
    assert not assets["SPY"].empty