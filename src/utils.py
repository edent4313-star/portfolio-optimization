import joblib
import os


class Utils:

    @staticmethod
    def save_model(model, path):

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(model, path)

        except Exception as e:
            raise RuntimeError(f"Save error: {e}")