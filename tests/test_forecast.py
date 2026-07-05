import numpy as np
import pandas as pd

from src.forecast import FutureForecaster


def test_create_future_dates():

    dates = FutureForecaster.create_future_dates(

        "2025-12-31",

        periods=30

    )

    assert len(dates) == 30


def test_trend_analysis():

    prices = np.array([100,110,120,130])

    trend = FutureForecaster.analyze_trend(prices)

    assert trend["Trend"] == "Upward"


def test_confidence_intervals():

    preds = np.array([100,101,102])

    returns = pd.Series([0.01,-0.02,0.03,0.01])

    lower, upper = FutureForecaster.confidence_intervals(

        preds,

        returns

    )

    assert len(lower) == len(preds)

    assert len(upper) == len(preds)


def test_market_analysis():

    trend = {

        "Trend":"Upward",

        "Percent Change":20

    }

    opp, risks = FutureForecaster.market_analysis(trend)

    assert len(opp) > 0

    assert len(risks) > 0