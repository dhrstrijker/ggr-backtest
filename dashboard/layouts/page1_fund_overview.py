"""Page 1: Fund Overview layout."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..components.metrics_card import create_metric_card
from .base import create_page_header


def create_fund_overview_layout(data_store) -> html.Div:
    """
    Create the Fund Overview page layout.

    Displays:
    - Header metrics (Total Return, YTD vs SPY, Active Pairs, Capital Usage)
    - Committed/Fully Invested toggle
    - Equity curve chart with SPY overlay
    - Risk metrics table
    """
    return html.Div([
        # Page header
        create_page_header(
            "Fund Overview",
            "Strategy Performance vs Benchmark"
        ),

        # Header metrics row
        dbc.Row([
            dbc.Col(create_metric_card("Total Return", "total-return-metric"), md=3, sm=6, className="mb-3"),
            dbc.Col(create_metric_card("YTD Return", "ytd-return-metric"), md=3, sm=6, className="mb-3"),
            dbc.Col(create_metric_card("Active Pairs", "active-pairs-metric"), md=3, sm=6, className="mb-3"),
            dbc.Col(create_metric_card("Capital Usage", "capital-usage-metric"), md=3, sm=6, className="mb-3"),
        ], className="mb-4"),

        # Return calculation toggle and mode indicator
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Return Calculation:", className="me-3 fw-bold"),
                                dbc.RadioItems(
                                    id="return-calc-toggle",
                                    options=[
                                        {"label": "Committed Capital", "value": "committed"},
                                        {"label": "Fully Invested", "value": "fully_invested"},
                                    ],
                                    value="committed",
                                    inline=True,
                                ),
                            ], md=6),
                            dbc.Col([
                                html.Div(id="current-mode-indicator", className="text-end"),
                            ], md=6),
                        ]),
                    ], className="py-2"),
                ], className="mb-4"),
            ], width=12),
        ]),

        # Main equity curve chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Equity Curve: GGR Strategy vs S&P 500", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="equity-curve-chart",
                            config={"displayModeBar": True, "scrollZoom": True},
                            style={"height": "450px"},
                        ),
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),

        # Risk metrics tables row
        dbc.Row([
            # Risk Metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Risk Metrics", className="mb-0"),
                    ]),
                    dbc.CardBody(id="risk-metrics-table"),
                ]),
            ], md=6, className="mb-3"),

            # Trade Statistics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Trade Statistics", className="mb-0"),
                    ]),
                    dbc.CardBody(id="trade-stats-table"),
                ]),
            ], md=6, className="mb-3"),
        ]),
    ])
