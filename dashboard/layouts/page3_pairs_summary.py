"""Page 3: Pairs Summary layout (Staggered Methodology)."""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

from .base import create_page_header


def create_pairs_summary_layout(data_store) -> html.Div:
    """
    Create the Pairs Summary page layout for staggered methodology.

    Displays:
    - Table with all pairs from all cycles, aggregated metrics
    - Number of cycles each pair was selected
    - Total P&L, win rate, average holding period
    - Clicking "Inspect" navigates to Pair Inspector for that pair
    """
    return html.Div([
        # Page header
        create_page_header(
            "Pairs Summary",
            "Aggregated performance across all portfolio cycles"
        ),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("All Pairs Performance", className="mb-0 d-inline"),
                        html.Span(
                            " (Aggregated across all cycles)",
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
