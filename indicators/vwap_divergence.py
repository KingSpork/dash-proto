from collections import deque
import numpy as np
import pandas as pd
from typing import Deque, Tuple


class VWAPDivergenceIndicator:
    def __init__(self, window: int = 21):
        self.window = int(window)  # Ensure window is always an integer
        self.bull_max: Deque[float] = deque(maxlen=self.window)
        self.bull_values: Deque[float] = deque(maxlen=self.window)
        self.bear_max: Deque[float] = deque(maxlen=self.window)
        self.bear_values: Deque[float] = deque(maxlen=self.window)

    def calculate_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validate required columns
        required_cols = {"High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Calculate market sessions
        df["is_market_session"] = (df.index.time >= pd.to_datetime("09:30").time()) & \
                                  (df.index.time <= pd.to_datetime("16:00").time())
        session_groups = (df["is_market_session"] & ~df["is_market_session"].shift(1).fillna(False)).cumsum()

        # Vectorized VWAP calculation
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        cum_tpv = (tp * df["Volume"]).groupby(session_groups).cumsum()
        cum_vol = df["Volume"].groupby(session_groups).cumsum()
        df["VWAP"] = cum_tpv / cum_vol

        # Calculate divergences
        df["bull_div"] = df["High"] - df["VWAP"]
        df["bear_div"] = df["VWAP"] - df["Low"]

        return df

    def update_history(self, df_day: pd.DataFrame):
        bull = df_day["bull_div"].max()
        bear = df_day["bear_div"].max()
        self.bull_max.append(bull)
        self.bull_values.append(bull)
        self.bear_max.append(bear)
        self.bear_values.append(bear)

    def get_current_bands(self) -> Tuple[float, float, float, float]:
        return (
            max(self.bull_max) if self.bull_max else 0,
            np.mean(self.bull_values) if self.bull_values else 0,
            max(self.bear_max) if self.bear_max else 0,
            np.mean(self.bear_values) if self.bear_values else 0
        )