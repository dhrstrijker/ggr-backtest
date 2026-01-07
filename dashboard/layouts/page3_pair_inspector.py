"""Page 3: Pair Inspector layout."""

from dash import html, dcc
import dash_bootstrap_components as dbc

from .base import create_page_header


def create_pair_inspector_layout(data_store) -> html.Div:
    """
    Create the Pair Inspector page layout.

    Displays:
    - Pair dropdown selector
    - Normalized prices chart (GGR Figure 1 style)
    - Distance chart with entry/exit bands
    - Pair statistics sidebar
    - Trade history table
    """
    # Create dropdown options from pairs
    pair_options = [
        {"label": f"{p[0]} / {p[1]}", "value": f"{p[0]}_{p[1]}"}
        for p in data_store.pairs
    ]

    return html.Div([
        # Page header
        create_page_header(
            "Pair Inspector",
            "Deep Dive Analysis for Selected Pair"
        ),

        dbc.Row([
            # Sidebar with selector and stats
            dbc.Col([
                # Pair selector
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Select Pair", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="pair-selector",
                            options=pair_options,
                            value=pair_options[0]["value"] if pair_options else None,
                            clearable=False,
                            className="mb-0",
                        ),
                    ]),
                ], className="mb-3"),

                # Pair statistics
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Pair Statistics", className="mb-0"),
                    ]),
                    dbc.CardBody(id="pair-stats-sidebar"),
                ]),
            ], md=3, className="mb-3"),

            # Main content area
            dbc.Col([
                # Chart A: Normalized Prices
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Normalized Prices", className="mb-0 d-inline"),
                        html.Span(
                            " (GGR Figure 1 style - $1 invested)",
                            className="text-muted small ms-2",
                        ),
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="normalized-prices-chart",
                            config={"displayModeBar": True, "scrollZoom": True},
                            style={"height": "350px"},
                        ),
                    ]),
                ], className="mb-3"),

                # Chart B: Distance with bands
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Spread Distance", className="mb-0 d-inline"),
                        html.Span(
                            " (Entry at ±2σ, Exit at 0)",
                            className="text-muted small ms-2",
                        ),
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="spread-distance-chart",
                            config={"displayModeBar": True, "scrollZoom": True},
                            style={"height": "300px"},
                        ),
                    ]),
                ]),
            ], md=9),
        ]),

        # Trade history table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Trade History", className="mb-0"),
                    ]),
                    dbc.CardBody(id="pair-trades-table"),
                ]),
            ], width=12),
        ], className="mt-4"),
    ])
