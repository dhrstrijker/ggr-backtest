"""Page 1: Fund Overview layout for staggered methodology."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..components.metrics_card import create_metric_card
from .base import create_page_header


def create_fund_overview_layout(data_store) -> html.Div:
    """
    Create the Fund Overview page layout for staggered methodology.

    Displays:
    - Header metrics (Annualized Return, Sharpe Ratio, Active Portfolios, Total Trades)
    - Cumulative returns chart (monthly)
    - Monthly returns bar chart
    - Risk metrics table
    - Trade Statistics table
    """
    return html.Div([
        # Page header
        create_page_header(
            "Fund Overview",
            "GGR Staggered Portfolio Performance"
        ),

        # Header metrics row
        dbc.Row([
            dbc.Col(create_metric_card("Annualized Return", "annualized-return-metric"), md=3, sm=6, className="mb-3"),
            dbc.Col(create_metric_card("Sharpe Ratio", "sharpe-ratio-metric"), md=3, sm=6, className="mb-3"),
            dbc.Col(create_metric_card("Avg Active Portfolios", "active-portfolios-metric"), md=3, sm=6, className="mb-3"),
            dbc.Col(create_metric_card("Total Trades", "total-trades-metric"), md=3, sm=6, className="mb-3"),
        ], className="mb-4"),

        # Mode indicator row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="current-mode-indicator"),
                            ], width=12),
                        ]),
                    ], className="py-2"),
                ], className="mb-4"),
            ], width=12),
        ]),

        # Cumulative returns chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Cumulative Returns (Monthly)", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="equity-curve-chart",
                            config={"displayModeBar": True, "scrollZoom": True},
                            style={"height": "400px"},
                        ),
                    ]),
                ]),
            ], width=12),
        ], className="mb-4"),

        # Monthly returns bar chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Monthly Returns", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="monthly-returns-chart",
                            config={"displayModeBar": True},
                            style={"height": "250px"},
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
