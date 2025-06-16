import os
import requests
import pandas as pd
import datetime
from dash import Dash, html, dcc, Input, Output, callback_context, State, no_update
from typing import Literal, Optional
import plotly.graph_objs as go
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")
assert API_KEY, "Please set your POLYGON_API_KEY in the environment."

ThemeType = Literal["light", "dark"]


def fetch_data(symbol: str, date: str, timespan: str = "minute", limit: int = 500) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/2025-06-12/2025-06-12"
    params = {"adjusted": "true", "sort": "asc", "limit": limit, "apiKey": API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json().get("results", [])
    if not data:
        raise ValueError("No results returned from Polygon API")

    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.rename(
        columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"},
        inplace=True,
    )
    df.set_index("t", inplace=True)

    df["TPV"] = (df["High"] + df["Low"] + df["Close"]) / 3 * df["Volume"]
    df["cum_TPV"] = df["TPV"].cumsum()
    df["cum_Volume"] = df["Volume"].cumsum()
    df["VWAP"] = df["cum_TPV"] / df["cum_Volume"]
    return df


def create_figure(df: pd.DataFrame, symbol: str, theme: ThemeType) -> go.Figure:
    layout_theme = {
        "plot_bgcolor": "#1e1e1e" if theme == "dark" else "#ffffff",
        "paper_bgcolor": "#1e1e1e" if theme == "dark" else "#ffffff",
        "font": {"color": "#ffffff" if theme == "dark" else "#000000"},
        "xaxis": {"title": "Time"},
        "yaxis": {"title": "Price"},
        "xaxis_rangeslider_visible": False,
        "margin": {"t": 40, "l": 50, "r": 50, "b": 40},
    }

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles",
            ),
            go.Scatter(
                x=df.index,
                y=df["VWAP"],
                mode="lines",
                name="VWAP",
                line=dict(color="orange", width=2),
            ),
        ],
        layout=go.Layout(title=f"{symbol} 1-Minute Chart with VWAP", **layout_theme),
    )
    return fig


# Initialize app and data
symbol = "QQQ"
today_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
df = fetch_data(symbol, today_str)

app = Dash(__name__)
app.title = f"{symbol} VWAP Chart"

app.layout = html.Div(
    [
        dcc.Store(id="theme-store", storage_type="local"),  # No initial data!
        html.Div(
            [
                html.Label("Theme:"),
                dcc.Dropdown(
                    id="theme-selector",
                    options=[
                        {"label": "Light", "value": "light"},
                        {"label": "Dark", "value": "dark"},
                    ],
                    value="light",  # Default UI value
                    clearable=False,
                    style={"width": "200px"},
                ),
            ],
            style={"padding": "10px"},
        ),
        html.Div(
            id="main-container",
            children=[
                html.H1(f"{symbol} Candlestick + VWAP"),
                dcc.Graph(id="chart", style={"height": "80vh", "width": "100%"}),
            ],
            style={"padding": "10px"},
        ),
    ]
)


@app.callback(
    Output("theme-store", "data"),
    Input("theme-selector", "value"),
    State("theme-store", "data"),
    prevent_initial_call=False,
)
def handle_theme(selector_value: ThemeType, stored_value: Optional[ThemeType]) -> ThemeType:
    ctx = callback_context
    if not ctx.triggered:
        # Initial load - use dropdown default
        return selector_value
    elif stored_value is None:
        # First interaction - store the selected value
        return selector_value
    else:
        # Normal operation - update store with new selection
        return selector_value


# Separate callback for chart updates
@app.callback(
    Output("chart", "figure"),
    Input("theme-store", "data"),
)
def update_chart(theme: Optional[ThemeType]) -> go.Figure:
    effective_theme: ThemeType = theme if theme is not None else "light"
    return create_figure(df, symbol, effective_theme)


if __name__ == "__main__":
    app.run(debug=True)
