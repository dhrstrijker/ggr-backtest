"""Page 2: Live Trading Monitor layout."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..components.metrics_card import create_metric_card
from .base import create_page_header


def create_live_monitor_layout(data_store) -> html.Div:
    """
    Create the Live Trading Monitor page layout.

    Displays:
    - Date slider to scrub through trading period
    - Summary metrics at selected date
    - Top pairs status table with color-coded spreads
    """
    # Get date range for slider
    dates = data_store.get_trading_dates()

    return html.Div([
        # Page header
        create_page_header(
            "Live Trading Monitor",
            "Position Status at Selected Date"
        ),

        # Date selector
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Select Date", className="mb-0"),
                            ], md=6),
                            dbc.Col([
                                html.Div(id="current-date-display", className="h5 mb-0 text-end"),
                            ], md=6),
                        ]),
                    ]),
                    dbc.CardBody([
                        dcc.Slider(
                            id="date-slider",
                            min=0,
                            max=len(dates) - 1,
                            value=len(dates) - 1,  # Default to latest
                            marks={
                                0: {"label": dates[0].strftime("%Y-%m-%d")},
                                len(dates) // 4: {"label": dates[len(dates) // 4].strftime("%Y-%m")},
                                len(dates) // 2: {"label": dates[len(dates) // 2].strftime("%Y-%m")},
                                3 * len(dates) // 4: {"label": dates[3 * len(dates) // 4].strftime("%Y-%m")},
                                len(dates) - 1: {"label": dates[-1].strftime("%Y-%m-%d")},
                            },
                            step=1,
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="mt-3",
                        ),
                        # Animation controls
                        html.Div([
                            dbc.ButtonGroup([
                                dbc.Button(
                                    html.I(className="bi bi-skip-start-fill"),
                                    id="reset-button",
                                    color="secondary",
                                    size="sm",
                                    title="Reset to start",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-play-fill"),
                                    id="play-button",
                                    color="primary",
                                    size="sm",
                                    title="Play animation",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-skip-end-fill"),
                                    id="end-button",
                                    color="secondary",
                                    size="sm",
                                    title="Jump to end",
                                ),
                            ], className="mt-3"),
                            dcc.Interval(
                                id="animation-interval",
                                interval=500,  # 0.5 second per day
                                disabled=True,
                            ),
                        ], className="text-center"),
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),

        # Summary metrics at selected date
        dbc.Row([
            dbc.Col(create_metric_card("Open Positions", "open-positions-count"), md=3, sm=6, className="mb-3"),
            dbc.Col(create_metric_card("Unrealized P&L", "unrealized-pnl"), md=3, sm=6, className="mb-3"),
            dbc.Col(create_metric_card("Realized P&L (YTD)", "realized-pnl"), md=3, sm=6, className="mb-3"),
            dbc.Col(create_metric_card("Signals Today", "signals-today"), md=3, sm=6, className="mb-3"),
        ], className="mb-4"),

        # Top pairs status table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Top Pairs Status", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        # Legend
                        html.Div([
                            dbc.Badge("OPEN", color="primary", className="me-2"),
                            html.Span("Position open", className="me-4 small"),
                            dbc.Badge("WAITING", color="secondary", className="me-2"),
                            html.Span("Awaiting entry signal", className="me-4 small"),
                            html.Span("|", className="me-4 text-muted"),
                            html.Span("Distance: ", className="small"),
                            html.Span(">2σ", className="text-danger me-2 small fw-bold"),
                            html.Span("0.5-2σ", className="text-warning me-2 small fw-bold"),
                            html.Span("<0.5σ", className="text-success small fw-bold"),
                        ], className="mb-3 pb-2 border-bottom"),
                        html.Div(id="pairs-status-table"),
                    ]),
                ]),
            ], width=12),
        ]),
    ])
