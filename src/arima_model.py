import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.arima.model import ARIMA


class TimeSeriesARIMA:

    def __init__(self):
        self.model = None
        self.results = None

    def grid_search(self, series, p=range(0,4), d=[0], q=range(0,4)):

        best_aic = np.inf
        best_order = None
        best_model = None

        try:

            for order in product(p, d, q):

                try:
                    model = ARIMA(series, order=order)
                    result = model.fit()

                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = order
                        best_model = result

                except:
                    continue

            self.results = best_model
            self.model = best_model.model

            return best_model, best_order

        except Exception as e:
            raise RuntimeError(f"ARIMA error: {e}")

    def forecast(self, steps):
        try:
            return self.results.forecast(steps=steps)
        except Exception as e:
            raise RuntimeError(f"Forecast error: {e}")