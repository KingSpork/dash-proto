import requests
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import pickle
import os
from pathlib import Path


class MarketSessionManager:
    def __init__(self, api_key: str, cache_dir: str = ".market_cache"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Core data structures
        self.holidays: Dict[date, str] = {}
        self.early_closes: Dict[date, time] = {}
        self.irregular_sessions: Dict[date, Tuple[Optional[time], Optional[time]]] = {}

        # Special cases (e.g., COVID, 9/11)
        self._load_historical_exceptions()
        self._load_polygon_calendar()

    def _load_historical_exceptions(self):
        """Hardcoded irregular sessions that APIs might miss"""
        # Format: {date: (open_time, close_time), None = closed}
        self.irregular_sessions.update(
            {
                # COVID-19 early closures
                date(2020, 3, 16): (time(9, 30), time(16, 0)),  # Volatility halt
                date(2020, 3, 20): (time(9, 30), time(15, 15)),  # Quarterly expiry
                # Historical closures
                date(2001, 9, 11): (None, None),  # 9/11
                date(2012, 10, 29): (None, None),  # Hurricane Sandy
                # Add more as needed...
            }
        )

    def _load_polygon_calendar(self, years: List[int] = None):
        """Load and cache multiple years of data"""
        years = years or [datetime.now().year, datetime.now().year + 1]
        for year in years:
            cache_file = self.cache_dir / f"nyse_calendar_{year}.pkl"

            if cache_file.exists():
                self._load_from_cache(cache_file)
            else:
                self._fetch_polygon_data(year)
                self._save_to_cache(cache_file)

    def _fetch_polygon_data(self, year: int):
        """Get official calendar from Polygon"""
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{year}?apiKey={self.api_key}"
        try:
            response = requests.get(url).json()
            for day in response["results"]:
                dt = pd.to_datetime(day["date"]).date()

                if day.get("status") == "closed":
                    self.holidays[dt] = day.get("name", "Holiday")
                elif day.get("change") == "Early close":
                    self.early_closes[dt] = pd.to_datetime(day["time"]).time()
        except Exception as e:
            print(f"Failed to fetch Polygon data: {e}")
            self._load_fallback_calendar(year)

    def _load_fallback_calendar(self, year: int):
        """Use backup data sources if Polygon fails"""
        try:
            import pandas_market_calendars as mcal

            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
            for dt in schedule.index:
                if "early" in schedule.loc[dt]["market_open"]:
                    self.early_closes[dt.date()] = schedule.loc[dt]["market_close"].time()
        except:
            print("Using hardcoded holidays as last resort")
            self._load_default_holidays(year)

    def get_session_bounds(self, dt: date) -> Tuple[Optional[time], Optional[time]]:
        """Returns (open_time, close_time) for a date"""
        if dt in self.irregular_sessions:
            return self.irregular_sessions[dt]

        if not self.is_trading_day(dt):
            return (None, None)

        return (time(9, 30), self.early_closes.get(dt, time(16, 0)))

    def get_session_mask(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized session detection"""

        def _check_time(row):
            dt = row.name.date()
            open_time, close_time = self.get_session_bounds(dt)

            if open_time is None:  # Market closed
                return False

            return row.name.time() >= open_time and row.name.time() <= close_time

        return df.apply(_check_time, axis=1)

    def get_session_shapes(self, start_date: date, end_date: date) -> List[dict]:
        """
        Generates Plotly shapes for shading non-market sessions
        Returns list of shape dictionaries for fig.add_shape()
        """
        shapes = []
        current_date = start_date

        while current_date <= end_date:
            open_time, close_time = self.get_session_bounds(current_date)

            # Skip fully closed days
            if open_time is None:
                current_date += timedelta(days=1)
                continue

            # Pre-market (gold)
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=datetime.combine(current_date, time(4, 0)),  # Pre-market starts ~4AM
                    x1=datetime.combine(current_date, open_time),
                    y0=0,
                    y1=1,
                    fillcolor="gold",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            )

            # Post-market (dark blue)
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=datetime.combine(current_date, close_time),
                    x1=datetime.combine(current_date, time(20, 0)),  # Post-market ends ~8PM
                    y0=0,
                    y1=1,
                    fillcolor="darkblue",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            )

            current_date += timedelta(days=1)

        return shapes
