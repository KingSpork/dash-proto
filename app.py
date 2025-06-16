import os
from datetime import datetime
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output
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

# --- Color Constants ---
COLOR_UP = 'rgb(8, 153, 129)'
COLOR_DOWN = 'rgb(242, 54, 69)'

# --- Data Initialization ---
def fetch_data(symbol: str, date: str) -> pd.DataFrame:
    """Fetch and standardize data from Polygon API"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/2025-06-12/2025-06-12"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}
    response = requests.get(url, params=params).json()

    df = pd.DataFrame(response["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("t")

    df = df.rename(columns={
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume"
    })
    df.index = pd.to_datetime(df.index, unit='ms').tz_localize('UTC').tz_convert('America/New_York')

    return df[["Open", "High", "Low", "Close", "Volume"]]

today = datetime.now().strftime("%Y-%m-%d")
df = fetch_data(symbol, today)

# --- Indicator Setup ---
session_mgr = MarketSessionManager(API_KEY)
divergence_indicator = VWAPDivergenceIndicator(window=21)

df = divergence_indicator.calculate_divergences(df)


# --- Helper functions ---
def create_wicks_traces(df, wick_width=1):
    x_vals = df.index
    l = df['Low'].values
    h = df['High'].values
    o = df['Open'].values
    c = df['Close'].values

    x_wicks_up = []
    y_wicks_up = []
    x_wicks_down = []
    y_wicks_down = []

    for xi, low, high, open_, close_ in zip(x_vals, l, h, o, c):
        if close_ >= open_:
            x_wicks_up.extend([xi, xi, None])
            y_wicks_up.extend([low, high, None])
        else:
            x_wicks_down.extend([xi, xi, None])
            y_wicks_down.extend([low, high, None])

    trace_up = go.Scatter(
        x=x_wicks_up,
        y=y_wicks_up,
        mode='lines',
        line=dict(color=COLOR_UP, width=wick_width),
        name='Wicks Up',
        hoverinfo='skip',
        showlegend=False
    )

    trace_down = go.Scatter(
        x=x_wicks_down,
        y=y_wicks_down,
        mode='lines',
        line=dict(color=COLOR_DOWN, width=wick_width),
        name='Wicks Down',
        hoverinfo='skip',
        showlegend=False
    )

    return trace_up, trace_down


def create_bodies_trace(df, candle_width_ms=60000):
    o = df['Open'].values
    c = df['Close'].values

    colors = [COLOR_UP if close_ >= open_ else COLOR_DOWN for open_, close_ in zip(o, c)]

    return go.Bar(
        x=df.index,
        y=np.abs(c - o),
        base=np.minimum(o, c),
        marker_color=colors,
        width=candle_width_ms,
        name='Bodies',
        hoverinfo='skip',
        showlegend=False
    )


# --- App Setup ---
app = Dash(__name__)
app.title = f"{symbol} VWAP Divergence"

app.layout = html.Div([
    dcc.Store(id="theme-store", storage_type="local"),
    dcc.Interval(id="update-interval", interval=60_000),

    html.Div([
        html.Label("Theme:"),
        dcc.Dropdown(
            id="theme-selector",
            options=[{"label": t, "value": t} for t in ["light", "dark"]],
            value="light",
            clearable=False
        ),
    ], style={"padding": "10px"}),

    dcc.Graph(
        id="chart",
        style={"height": "80vh"},
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': [
                'zoom2d',
                'pan2d',
                'zoomIn2d',
                'zoomOut2d',
                'autoScale2d',
                'resetScale2d'
            ]
        }
    )
])


# --- Callbacks ---
@app.callback(
    Output("chart", "figure"),
    [Input("update-interval", "n_intervals"),
     Input("theme-selector", "value")]
)
def update_chart(n_intervals, theme):
    global df

    fig = go.Figure()

    trace_wicks_up, trace_wicks_down = create_wicks_traces(df, wick_width=0.5)
    fig.add_trace(trace_wicks_up)
    fig.add_trace(trace_wicks_down)
    fig.add_trace(create_bodies_trace(df))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["VWAP"],
        line=dict(color="orange", width=2),
        name="VWAP"
    ))

    start_date = df.index.min().date()
    end_date = df.index.max().date()
    for shape in session_mgr.get_session_shapes(start_date, end_date):
        fig.add_shape(shape)

    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        margin=dict(t=10, b=40, l=50, r=50),
        height=600,
        dragmode="pan",
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=False),
            tickformat='%H:%M',
        ),
        yaxis=dict(showgrid=True),
        xaxis_fixedrange=False,
        yaxis_fixedrange=False,
    )

    fig.update_yaxes(
        side='right',
        tickfont=dict(size=10),
        fixedrange=False
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
