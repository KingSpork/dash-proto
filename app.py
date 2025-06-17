import os
from datetime import datetime
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, ctx
import plotly.graph_objects as go
from dotenv import load_dotenv
import requests
import json
from typing import Literal, Optional, Dict, Any

# Import custom indicators
from indicators.market_sessions import MarketSessionManager
from indicators.vwap_divergence import VWAPDivergenceIndicator

# --- Constants ---
COLOR_UP = 'rgb(8, 153, 129)'
COLOR_DOWN = 'rgb(242, 54, 69)'
CONFIG_FILE = 'config.json'
TIMEFRAME_MAP: Dict[str, Dict[str, Any]] = {
    '1m': {'timespan': 'minute', 'multiplier': 1, 'width_ms': 45_000},
    '5m': {'timespan': 'minute', 'multiplier': 5, 'width_ms': 225_000},
    '15m': {'timespan': 'minute', 'multiplier': 15, 'width_ms': 675_000},
    '1h': {'timespan': 'hour', 'multiplier': 1, 'width_ms': 2_700_000},
    '1D': {'timespan': 'day', 'multiplier': 1, 'width_ms': 86_400_000}
}

# --- Type Definitions ---
Timeframe = Literal['1m', '5m', '15m', '1h', '1D']

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")
symbol = "QQQ"


# --- Data Initialization ---
def fetch_data(symbol: str, date: str, timeframe: Timeframe = '1m') -> Optional[pd.DataFrame]:
    """Fetch market data for given symbol, date and timeframe."""
    try:
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": API_KEY
        }

        tf_config = TIMEFRAME_MAP[timeframe]
        url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
               f"{tf_config['multiplier']}/{tf_config['timespan']}/2025-06-12/2025-06-12")

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if 'results' not in data or not data['results']:
            return None

        df = pd.DataFrame(data["results"])
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("t")
        df = df.rename(columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume"
        })

        # Only convert timezone for intraday data
        if timeframe != '1D':
            df.index = pd.to_datetime(df.index, unit='ms').tz_localize('UTC').tz_convert('America/New_York')

        return df[["Open", "High", "Low", "Close", "Volume"]]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing data: {e}")
        return None


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"theme": "light", "timeframe": "1m"}


def save_config(config: Dict[str, Any]) -> None:
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)


# --- Initial Setup ---
today = datetime.now().strftime("%Y-%m-%d")
initial_timeframe: Timeframe = '1m'
df = fetch_data(symbol, today, initial_timeframe)
if df is None:
    raise ValueError("Failed to fetch initial data")

session_mgr = MarketSessionManager(API_KEY)
divergence_indicator = VWAPDivergenceIndicator(window=21)
df = divergence_indicator.calculate_divergences(df)


# --- Helper Functions ---
def create_wicks_traces(df: pd.DataFrame, wick_width: float = 1) -> tuple[go.Scatter, go.Scatter]:
    x_vals = df.index
    l = df['Low'].values
    h = df['High'].values
    o = df['Open'].values
    c = df['Close'].values

    x_wicks_up, y_wicks_up = [], []
    x_wicks_down, y_wicks_down = [], []

    for xi, low, high, open_, close_ in zip(x_vals, l, h, o, c):
        if close_ >= open_:
            x_wicks_up.extend([xi, xi, None])
            y_wicks_up.extend([low, high, None])
        else:
            x_wicks_down.extend([xi, xi, None])
            y_wicks_down.extend([low, high, None])

    return (
        go.Scatter(
            x=x_wicks_up,
            y=y_wicks_up,
            mode='lines',
            line=dict(color=COLOR_UP, width=wick_width),
            showlegend=False
        ),
        go.Scatter(
            x=x_wicks_down,
            y=y_wicks_down,
            mode='lines',
            line=dict(color=COLOR_DOWN, width=wick_width),
            showlegend=False
        )
    )


def create_bodies_trace(df: pd.DataFrame, timeframe: Timeframe) -> go.Bar:
    o = df['Open'].values
    c = df['Close'].values
    colors = [COLOR_UP if close_ >= open_ else COLOR_DOWN for open_, close_ in zip(o, c)]
    return go.Bar(
        x=df.index,
        y=np.abs(c - o),
        base=np.minimum(o, c),
        marker_color=colors,
        width=TIMEFRAME_MAP[timeframe]['width_ms'],
        showlegend=False
    )


# --- App Setup ---
app = Dash(__name__)
app.title = f"{symbol} VWAP Divergence"

initial_config = load_config()

app.layout = html.Div([
    dcc.Store(id="theme-store", storage_type="local", data=initial_config["theme"]),
    dcc.Store(id="timeframe-store", storage_type="local", data=initial_config.get("timeframe", "1m")),
    dcc.Interval(id="update-interval", interval=60_000),

    html.Div([
        html.Label("Theme:"),
        dcc.Dropdown(
            id="theme-selector",
            options=[{"label": t, "value": t} for t in ["light", "dark"]],
            value=initial_config["theme"],
            clearable=False,
            style={"width": "100px"}
        ),
        html.Label("Timeframe:"),
        dcc.Dropdown(
            id="timeframe-selector",
            options=[
                {"label": "1 Minute", "value": "1m"},
                {"label": "5 Minutes", "value": "5m"},
                {"label": "15 Minutes", "value": "15m"},
                {"label": "1 Hour", "value": "1h"},
                {"label": "1 Day", "value": "1D"}
            ],
            value=initial_config.get("timeframe", "1m"),
            clearable=False,
            style={"width": "120px"}
        ),
        html.Button("Save Config", id="save-config-btn", n_clicks=0),
        html.Button("Load Config", id="load-config-btn", n_clicks=0),
    ], style={
        "padding": "10px",
        "display": "flex",
        "gap": "10px",
        "alignItems": "center",
        "flexWrap": "wrap"
    }),

    dcc.Graph(
        id="chart",
        style={"height": "80vh", "width": "100%"},
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': [
                'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'
            ]
        }
    )
])


# --- Callbacks ---
@app.callback(
    Output("theme-store", "data"),
    [Input("theme-selector", "value"),
     Input("save-config-btn", "n_clicks")],
    [State("theme-selector", "value"),
     State("timeframe-selector", "value")],
    prevent_initial_call=True
)
def update_theme_or_save(
        theme_value: str,
        save_clicks: int,
        theme_state: str,
        timeframe_state: str
) -> str:
    trigger_id = ctx.triggered_id

    if trigger_id == "save-config-btn":
        config = {"theme": theme_state, "timeframe": timeframe_state}
        save_config(config)
        return theme_state
    elif trigger_id == "theme-selector":
        return theme_value
    return dash.no_update


@app.callback(
    Output("timeframe-store", "data"),
    Input("timeframe-selector", "value"),
    prevent_initial_call=True
)
def update_timeframe(timeframe: Timeframe) -> Timeframe:
    return timeframe


@app.callback(
    Output("chart", "figure"),
    [Input("update-interval", "n_intervals"),
     Input("theme-store", "data"),
     Input("timeframe-store", "data")],
    prevent_initial_call=True
)
def update_chart(n_intervals: int, theme: str, timeframe: Timeframe) -> go.Figure:
    global df

    if ctx.triggered_id == "timeframe-store":
        new_data = fetch_data(symbol, today, timeframe)
        if new_data is not None:
            df = new_data
            df = divergence_indicator.calculate_divergences(df)
        else:
            return go.Figure()  # Return empty figure on error

    # Create figure with consistent dimensions
    fig = go.Figure(
        layout={
            'height': 600,
            'margin': dict(t=10, b=40, l=50, r=50)
        }
    )

    trace_wicks_up, trace_wicks_down = create_wicks_traces(df, wick_width=0.5)
    fig.add_trace(trace_wicks_up)
    fig.add_trace(trace_wicks_down)
    fig.add_trace(create_bodies_trace(df, timeframe))

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

    # Update layout while maintaining dimensions
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        dragmode="pan",
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=False),
            tickformat='%H:%M' if timeframe != '1D' else '%Y-%m-%d',
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across",
            spikedash="dot",
            spikethickness=0.5,
        ),
        yaxis=dict(
            showgrid=True,
            spikecolor="white",
            spikemode="across",
            spikedash="dot",
            spikethickness=0.5,
        ),
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
    [Output("theme-selector", "value"),
     Output("timeframe-selector", "value")],
    Input("load-config-btn", "n_clicks"),
    prevent_initial_call=True
)
def load_config(n_clicks: int) -> tuple[str, str]:
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            return (
                config.get("theme", "light"),
                config.get("timeframe", "1m")
            )
    except Exception:
        return "light", "1m"


# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)