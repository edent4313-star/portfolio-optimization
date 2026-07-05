from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


class TimeSeriesLSTM:

    def build_model(self, input_shape):

        try:

            model = Sequential()

            model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))

            model.add(LSTM(32))
            model.add(Dropout(0.2))

            model.add(Dense(16, activation="relu"))
            model.add(Dense(1))

            model.compile(
                optimizer=Adam(0.001),
                loss="mse"
            )

            return model

        except Exception as e:
            raise RuntimeError(f"LSTM build error: {e}")