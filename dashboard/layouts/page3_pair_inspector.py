"""Page 3: Pair Inspector layout."""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_pair_inspector_layout(data_store) -> html.Div:
    """
    Create the Pair Inspector page layout.

    Displays:
    - Back to summary link
    - Dynamic pair title (from URL)
    - Normalized prices chart (GGR Figure 1 style)
    - Distance chart with entry/exit bands
    - Pair statistics sidebar
    - Trade history table
    """
    return html.Div([
        # Store for current pair (populated from URL by callback)
        dcc.Store(id="current-pair"),

        # Back link and header
        dbc.Row([
            dbc.Col([
                dcc.Link(
                    html.Span(["← Back to Pairs Summary"]),
                    href="/summary",
                    className="text-decoration-none",
                ),
            ], width="auto"),
        ], className="mb-2"),

        # Dynamic page header (pair name filled by callback)
        dbc.Row([
            dbc.Col([
                html.H4(id="pair-inspector-title", className="mb-0"),
                html.P("Deep Dive Analysis", className="text-muted mb-0"),
            ], width=12),
        ], className="mb-4"),

        dbc.Row([
            # Sidebar with stats
            dbc.Col([
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
                        html.Div([
                            html.Div([
                                html.H5("Spread Distance", className="mb-0 d-inline"),
                                html.Span(
                                    " (Entry at ±2σ, Exit at 0)",
                                    className="text-muted small ms-2",
                                ),
                            ], className="d-inline-block"),
                            html.Div([
                                html.Label("Cycle:", className="me-2 small"),
                                dcc.Dropdown(
                                    id="cycle-selector",
                                    placeholder="Select cycle...",
                                    clearable=False,
                                    style={"width": "280px", "display": "inline-block"},
                                ),
                            ], className="d-inline-flex align-items-center float-end"),
                        ], className="d-flex justify-content-between align-items-center"),
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
