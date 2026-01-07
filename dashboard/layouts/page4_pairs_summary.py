"""Page 4: Pairs Summary layout."""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

from .base import create_page_header


def create_pairs_summary_layout(data_store) -> html.Div:
    """
    Create the Pairs Summary page layout.

    Displays:
    - Table with all pairs, their P&L, number of trades
    - Clicking a row navigates to Pair Inspector for that pair
    """
    return html.Div([
        # Page header
        create_page_header(
            "Pairs Summary",
            "Overview of all trading pairs - Click a row to inspect"
        ),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("All Pairs Performance", className="mb-0 d-inline"),
                        html.Span(
                            " (Click row to inspect pair)",
                            className="text-muted small ms-2",
                        ),
                    ]),
                    dbc.CardBody([
                        html.Div(id="pairs-summary-table"),
                    ]),
                ]),
            ], width=12),
        ]),
    ])
