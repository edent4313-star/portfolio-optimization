import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Evaluator:

    @staticmethod
    def metrics(y_true, y_pred):

        try:
            epsilon = 1e-8

            mae = mean_absolute_error(y_true, y_pred)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            mape = np.mean(
                np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))
            ) * 100

            return mae, rmse, mape

        except Exception as e:
            raise RuntimeError(f"Evaluation error: {e}")