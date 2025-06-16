import requests
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import pickle
from pathlib import Path


class MarketSessionManager:
    def __init__(self, api_key: str, cache_dir: str = ".market_cache"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.holidays: Dict[date, str] = {}
        self.early_closes: Dict[date, time] = {}
        self.irregular_sessions: Dict[date, Tuple[Optional[time], Optional[time]]] = {}
        self._load_historical_exceptions()
        self._load_polygon_calendar()

    def _load_historical_exceptions(self):
        """Hardcoded irregular sessions"""
        self.irregular_sessions.update({
            date(2020, 3, 16): (time(9, 30), time(16, 0)),
            date(2020, 3, 20): (time(9, 30), time(15, 15)),
            date(2001, 9, 11): (None, None),
            date(2012, 10, 29): (None, None),
        })

    def _load_default_holidays(self, year: int):
        """Fallback hardcoded holidays"""
        holidays = {
            date(year, 1, 1): "New Year's Day",
            date(year, 7, 4): "Independence Day",
            date(year, 12, 25): "Christmas Day"
            # Add more as needed
        }
        self.holidays.update(holidays)

    def _save_to_cache(self, cache_file: Path, year: int):
        """Save calendar data to disk"""
        with open(cache_file, "wb") as f:
            pickle.dump({
                "holidays": {k: v for k, v in self.holidays.items() if k.year == year},
                "early_closes": {k: v for k, v in self.early_closes.items() if k.year == year}
            }, f)

    def _load_polygon_calendar(self, years: List[int] = None):
        """Load calendar data from Polygon or cache"""
        years = years or [datetime.now().year]
        for year in years:
            cache_file = self.cache_dir / f"nyse_calendar_{year}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    self.holidays.update(data["holidays"])
                    self.early_closes.update(data["early_closes"])
            else:
                self._fetch_polygon_data(year)
                self._save_to_cache(cache_file, year)

    def _fetch_polygon_data(self, year: int):
        """Fetch data from Polygon API"""
        url = f"https://api.polygon.io/v1/marketstatus/upcoming?apiKey={self.api_key}"
        try:
            response = requests.get(url).json()
            for day in response:
                dt = pd.to_datetime(day["date"]).date()
                if day["status"] == "closed":
                    self.holidays[dt] = day.get("name", "Holiday")
                elif day.get("change") == "Early close":
                    self.early_closes[dt] = pd.to_datetime(day["time"]).time()
        except:
            self._load_default_holidays(year)

    def get_session_bounds(self, dt: date) -> Tuple[Optional[time], Optional[time]]:
        """Get open/close times for a date"""
        if dt in self.irregular_sessions:
            return self.irregular_sessions[dt]
        if dt.weekday() >= 5 or dt in self.holidays:
            return (None, None)
        return (time(9, 30), self.early_closes.get(dt, time(16, 0)))

    def get_session_mask(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask for trading sessions"""
        return df.index.to_series().apply(
            lambda x: (x.date() not in self.holidays) and
                      (x.time() >= time(9, 30)) and
                      (x.time() <= self.early_closes.get(x.date(), time(16, 0)))
        )

    def get_session_shapes(self, start_date: date, end_date: date) -> List[dict]:
        """Generate Plotly shapes for session shading"""
        shapes = []
        current_date = start_date
        while current_date <= end_date:
            open_time, close_time = self.get_session_bounds(current_date)
            if open_time is None:
                current_date += timedelta(days=1)
                continue

            # Pre-market
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=datetime.combine(current_date, time(4, 0)),
                x1=datetime.combine(current_date, open_time),
                y0=0, y1=1, fillcolor="gold", opacity=0.2, layer="below", line_width=0
            ))

            # Post-market
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=datetime.combine(current_date, close_time),
                x1=datetime.combine(current_date, time(20, 0)),
                y0=0, y1=1, fillcolor="darkblue", opacity=0.2, layer="below", line_width=0
            ))

            current_date += timedelta(days=1)
        return shapes