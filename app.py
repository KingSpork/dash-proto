import os
from datetime import datetime, time
import pandas as pd
from dash import Dash, html, dcc, Input, Output, no_update
import plotly.graph_objects as go
from dotenv import load_dotenv
import requests


# Import custom indicators
from indicators.market_sessions import MarketSessionManager
from indicators.vwap_divergence import VWAPDivergenceIndicator

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")
symbol = "QQQ"


# --- Data Initialization ---
def fetch_data(symbol: str, date: str) -> pd.DataFrame:
    """Fetch and standardize data from Polygon API"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/2025-06-12/2025-06-12"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}
    response = requests.get(url, params=params).json()

    df = pd.DataFrame(response["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("t")

    # Standardize column names
    df = df.rename(columns={
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume"
    })

    return df[["Open", "High", "Low", "Close", "Volume"]]  # Explicit column order


today = datetime.now().strftime("%Y-%m-%d")
df = fetch_data(symbol, today)

# --- Indicator Setup ---
session_mgr = MarketSessionManager(API_KEY)
divergence_indicator = VWAPDivergenceIndicator(window=21)

# Process initial data
df = divergence_indicator.calculate_divergences(df)

# --- App Layout ---
app = Dash(__name__)
app.title = f"{symbol} VWAP Divergence"

app.layout = html.Div([
    dcc.Store(id="theme-store", storage_type="local"),
    dcc.Interval(id="update-interval", interval=60_000),  # 1 minute updates

    html.Div([
        html.Label("Theme:"),
        dcc.Dropdown(
            id="theme-selector",
            options=[{"label": t, "value": t} for t in ["light", "dark"]],
            value="light",
            clearable=False
        ),
    ], style={"padding": "10px"}),

    dcc.Graph(id="chart", style={"height": "80vh"})
])


# --- Callbacks ---
@app.callback(
    Output("chart", "figure"),
    [Input("update-interval", "n_intervals"),
     Input("theme-selector", "value")]
)
def update_chart(n_intervals, theme):
    # Get current data (in a real app, you'd refresh this)
    global df

    # Create base figure
    fig = go.Figure()

    # Add price data
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    # Add VWAP and divergence bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df["VWAP"],
        line=dict(color="orange", width=2),
        name="VWAP"
    ))

    # Add session shading
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    for shape in session_mgr.get_session_shapes(start_date, end_date):
        fig.add_shape(shape)

    # Apply theme
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        xaxis_rangeslider_visible=False
    )

    return fig


@app.callback(
    Output("theme-store", "data"),
    Input("theme-selector", "value"),
    prevent_initial_call=True
)
def update_theme_store(theme):
    return theme


# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
